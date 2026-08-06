"""OpenPI-style policy client modified for UAV-safe remote inference."""

from __future__ import annotations

import itertools
import logging
import threading
import time
from typing import Any

from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException
from websockets.sync.client import ClientConnection, connect

from uav_openpi import msgpack_numpy, protocol
from uav_openpi.base_policy import BasePolicy

logger = logging.getLogger(__name__)


class RemotePolicyError(RuntimeError):
    """Base class for remote policy failures that require a local safe action."""


class InferenceTimeout(RemotePolicyError):
    """No fresh policy result arrived before the local deadline."""


class WebsocketClientPolicy(BasePolicy):
    """Implements ``BasePolicy`` through a persistent WebSocket connection.

    This retains OpenPI's small client API while adding request envelopes,
    bounded waits, explicit reset, structured errors, and reconnect-on-next-call.
    A failed inference is never automatically replayed because its observation
    may already be stale for flight control.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | None = 8000,
        api_key: str | None = None,
        *,
        connect_timeout_s: float = 30.0,
        inference_timeout_s: float = 1.0,
        retry_interval_s: float = 1.0,
        max_message_bytes: int = 16 * 1024 * 1024,
        episode_id: str = "default",
    ) -> None:
        if host.startswith(("ws://", "wss://")):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._api_key = api_key
        self._connect_timeout_s = connect_timeout_s
        self._inference_timeout_s = inference_timeout_s
        self._retry_interval_s = retry_interval_s
        self._max_message_bytes = max_message_bytes
        self._episode_id = episode_id
        self._request_ids = itertools.count(1)
        self._packer = msgpack_numpy.Packer()
        self._lock = threading.Lock()
        self._ws: ClientConnection | None = None
        self._server_metadata: dict[str, Any] = {}
        self._warned_missing_goal_reached = False
        self._connect()

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._server_metadata)

    def get_server_metadata(self) -> dict[str, Any]:
        """OpenPI-compatible metadata accessor."""
        return self.metadata

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._ensure_connected()
            request_id = next(self._request_ids)
            request = protocol.make_infer_request(
                obs,
                request_id=request_id,
                episode_id=self._episode_id,
                deadline_ms=max(1, int(self._inference_timeout_s * 1000)),
            )
            started = time.monotonic()
            response = self._round_trip(request, timeout_s=self._inference_timeout_s)
            response = protocol.validate_response(response, request_id=request_id)
            self._raise_remote_error(response)
            result = response.get("result")
            if not isinstance(result, dict):
                raise protocol.ProtocolError("Successful inference response has no result dictionary")
            result = dict(result)
            if "goal_reached" not in result:
                if not self._warned_missing_goal_reached:
                    logger.warning("Server result omitted goal_reached; treating it as false for compatibility")
                    self._warned_missing_goal_reached = True
                result["goal_reached"] = False
            protocol.validate_inference_result(result)
            result["server_timing"] = response.get("server_timing", {})
            result["transport_timing"] = {"round_trip_ms": (time.monotonic() - started) * 1000}
            return result

    def reset(self, episode_id: str | None = None) -> None:
        with self._lock:
            if episode_id is not None:
                if not episode_id:
                    raise ValueError("episode_id must be non-empty")
                self._episode_id = episode_id
            self._ensure_connected()
            request_id = next(self._request_ids)
            request = protocol.make_reset_request(request_id=request_id, episode_id=self._episode_id)
            response = self._round_trip(request, timeout_s=self._inference_timeout_s)
            response = protocol.validate_response(response, request_id=request_id)
            self._raise_remote_error(response)

    def close(self) -> None:
        with self._lock:
            self._disconnect()

    def __enter__(self) -> "WebsocketClientPolicy":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _connect(self) -> None:
        deadline = time.monotonic() + self._connect_timeout_s
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        logger.info("Waiting for policy server at %s", self._uri)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                remaining = max(0.1, deadline - time.monotonic())
                self._ws = connect(
                    self._uri,
                    compression=None,
                    max_size=self._max_message_bytes,
                    additional_headers=headers,
                    proxy=None,
                    open_timeout=min(5.0, remaining),
                    close_timeout=1.0,
                    ping_interval=10.0,
                    ping_timeout=5.0,
                )
                raw_metadata = self._ws.recv(timeout=min(5.0, remaining))
                if isinstance(raw_metadata, str):
                    raise protocol.ProtocolError("Server metadata must use a binary frame")
                self._server_metadata = protocol.validate_metadata(msgpack_numpy.unpackb(raw_metadata))
                logger.info("Connected to policy server: %s", self._server_metadata)
                return
            except InvalidStatus as exc:
                self._disconnect()
                raise RemotePolicyError(f"Policy server rejected the WebSocket handshake: {exc}") from exc
            except (OSError, TimeoutError, ConnectionClosed, WebSocketException, protocol.ProtocolError) as exc:
                last_error = exc
                self._disconnect()
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(self._retry_interval_s, remaining))
        raise RemotePolicyError(f"Unable to connect to {self._uri}: {last_error}")

    def _ensure_connected(self) -> None:
        if self._ws is None:
            self._connect()

    def _round_trip(self, request: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        assert self._ws is not None
        try:
            self._ws.send(self._packer.pack(request))
            raw_response = self._ws.recv(timeout=timeout_s)
        except TimeoutError as exc:
            self._disconnect()
            raise InferenceTimeout(f"Policy server exceeded {timeout_s:.3f}s deadline") from exc
        except (OSError, ConnectionClosed) as exc:
            self._disconnect()
            raise RemotePolicyError(f"Policy server connection failed: {exc}") from exc
        if isinstance(raw_response, str):
            raise protocol.ProtocolError("Policy server response must use a binary frame")
        return msgpack_numpy.unpackb(raw_response)

    def _disconnect(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.debug("Ignoring error while closing WebSocket", exc_info=True)

    @staticmethod
    def _raise_remote_error(response: dict[str, Any]) -> None:
        if response.get("status") == "ok":
            return
        error = response.get("error", {})
        raise RemotePolicyError(f"{error.get('code', 'remote_error')}: {error.get('message', 'unknown error')}")
