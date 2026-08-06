"""OpenPI WebSocket policy server modified for the UAV protocol."""

from __future__ import annotations

import asyncio
import http
import logging
import time
from typing import Any

import numpy as np
import websockets
import websockets.asyncio.server as websocket_server

from uav_openpi import msgpack_numpy, protocol
from uav_openpi.base_policy import BasePolicy
from uav_openpi.observation_recorder import ObservationRecorder

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serve a ``BasePolicy`` using the OpenPI request-response pattern."""

    def __init__(
        self,
        policy: BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict[str, Any] | None = None,
        *,
        max_message_bytes: int = 16 * 1024 * 1024,
        api_key: str | None = None,
        verbose_wire: bool = False,
        observation_recorder: ObservationRecorder | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._max_message_bytes = max_message_bytes
        self._api_key = api_key
        self._verbose_wire = bool(verbose_wire)
        self._observation_recorder = observation_recorder
        self._metadata = protocol.default_metadata()
        if self._verbose_wire:
            self._metadata.update(policy.metadata)
        if metadata:
            self._metadata.update(metadata)
        self._metadata["protocol"] = protocol.PROTOCOL_NAME
        self._metadata["protocol_version"] = protocol.PROTOCOL_VERSION
        self._metadata["result_schema"] = {
            "goal_reached": "bool",
            "actions_when_not_reached": "float32[1,4] body_flu local_pose_delta",
        }
        self._infer_lock: asyncio.Lock | None = None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            logger.info("Policy server stopped")
        finally:
            if self._observation_recorder is not None:
                self._observation_recorder.close()

    async def run(self) -> None:
        self._infer_lock = asyncio.Lock()
        async with websocket_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=self._max_message_bytes,
            process_request=self._process_request,
            ping_interval=3000.0,
            ping_timeout=3000.0,
        ) as server:
            logger.info("Policy server listening on ws://%s:%s", self._host, self._port)
            await server.serve_forever()

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        packed_metadata = packer.pack(self._metadata)
        await websocket.send(packed_metadata)
        logger.info("Sent metadata to %s bytes=%s", websocket.remote_address, len(packed_metadata))
        while True:
            request: dict[str, Any] | None = None
            try:
                raw_request = await websocket.recv()
                if isinstance(raw_request, str):
                    raise protocol.ProtocolError("Requests must use binary WebSocket frames")
                request = protocol.validate_request(msgpack_numpy.unpackb(raw_request))
                response = await self._dispatch(request)
                packed_response = packer.pack(response)
                await websocket.send(packed_response)
                logger.info(
                    "Wire transfer op=%s request_id=%s episode_id=%s request_bytes=%s response_bytes=%s",
                    request["op"],
                    request["request_id"],
                    request["episode_id"],
                    len(raw_request),
                    len(packed_response),
                )
            except websockets.ConnectionClosed as exc:
                logger.info(
                    "Connection from %s closed code=%s reason=%s",
                    websocket.remote_address,
                    exc.code,
                    exc.reason or "<none>",
                )
                return
            except protocol.ProtocolError as exc:
                await websocket.send(
                    packer.pack(protocol.make_error_response(request, code="invalid_request", message=str(exc)))
                )
            except Exception:
                logger.exception("Policy request failed")
                await websocket.send(
                    packer.pack(
                        protocol.make_error_response(
                            request,
                            code="internal_error",
                            message="Policy inference failed; inspect server logs",
                        )
                    )
                )

    async def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        if request["op"] == "reset":
            assert self._infer_lock is not None
            async with self._infer_lock:
                self._policy.reset(request["episode_id"])
            return protocol.make_success_response(request)

        assert self._infer_lock is not None
        observation_received_ns = time.time_ns()
        observation_recorded = False
        if self._observation_recorder is not None:
            try:
                observation_recorded = self._observation_recorder.record(
                    request["observation"],
                    request_id=request["request_id"],
                    episode_id=request["episode_id"],
                    received_ns=observation_received_ns,
                )
            except Exception:
                logger.exception(
                    "Failed to queue observation recording; continuing inference request_id=%s episode_id=%s",
                    request["request_id"],
                    request["episode_id"],
                )
        started = time.monotonic()
        async with self._infer_lock:
            queue_ms = (time.monotonic() - started) * 1000
            if protocol.request_deadline_expired(request, elapsed_ms=queue_ms):
                return protocol.make_error_response(
                    request,
                    code="deadline_exceeded",
                    message="Request exceeded its server queue deadline",
                )
            infer_started = time.monotonic()
            result = await asyncio.to_thread(self._policy.infer, request["observation"])
            infer_ms = (time.monotonic() - infer_started) * 1000
        if not isinstance(result, dict):
            raise TypeError("Policy.infer() must return a dictionary")
        try:
            protocol.validate_inference_result(result)
        except protocol.ProtocolError as exc:
            raise TypeError(f"Policy returned an invalid inference result: {exc}") from exc
        wire_result = _make_wire_result(result, include_debug=self._verbose_wire)
        if observation_recorded and self._observation_recorder is not None:
            try:
                self._observation_recorder.record_inference(
                    result,
                    request_id=request["request_id"],
                    episode_id=request["episode_id"],
                    received_ns=observation_received_ns,
                    completed_ns=time.time_ns(),
                    queue_ms=queue_ms,
                    infer_ms=infer_ms,
                )
            except Exception:
                logger.exception(
                    "Failed to queue inference recording; continuing response request_id=%s episode_id=%s",
                    request["request_id"],
                    request["episode_id"],
                )
        logger.info(
            "Handled %s request_id=%s episode_id=%s queue_ms=%.2f infer_ms=%.2f action_shape=%s",
            request["op"],
            request["request_id"],
            request["episode_id"],
            queue_ms,
            infer_ms,
            _shape_of(result.get("actions")),
        )
        payload: dict[str, Any] = {"result": wire_result}
        if self._verbose_wire:
            payload["server_timing"] = {"queue_ms": queue_ms, "infer_ms": infer_ms}
        return protocol.make_success_response(request, **payload)

    def _process_request(
        self,
        connection: websocket_server.ServerConnection,
        request: websocket_server.Request,
    ) -> websocket_server.Response | None:
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        if self._api_key and request.headers.get("Authorization") != f"Api-Key {self._api_key}":
            return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Unauthorized\n")
        return None


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def _make_wire_result(result: dict[str, Any], *, include_debug: bool) -> dict[str, Any]:
    """Build the small, language-neutral protocol-v2 result sent to the aircraft."""
    wire_result: dict[str, Any] = {"goal_reached": result["goal_reached"]}
    if not result["goal_reached"]:
        actions = np.asarray(result["actions"], dtype=np.float32)
        wire_result["actions"] = actions.astype(float).tolist()
    if include_debug and "debug" in result:
        wire_result["debug"] = result["debug"]
    return wire_result
