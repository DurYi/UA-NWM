"""Versioned UAV messages carried by the OpenPI-style WebSocket transport."""

from __future__ import annotations

import time
from typing import Any


PROTOCOL_NAME = "uav-openpi"
PROTOCOL_VERSION = 2
SUPPORTED_OPERATIONS = frozenset({"infer", "reset"})


class ProtocolError(RuntimeError):
    """The peer sent a malformed or incompatible protocol message."""


def default_metadata() -> dict[str, Any]:
    return {
        "service": "uav-openpi-policy-server",
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
    }


def make_infer_request(
    observation: dict[str, Any],
    *,
    request_id: int,
    episode_id: str,
    deadline_ms: int,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "op": "infer",
        "request_id": request_id,
        "episode_id": episode_id,
        "created_ns": time.time_ns(),
        "deadline_ms": deadline_ms,
        "observation": observation,
    }


def make_reset_request(*, request_id: int, episode_id: str) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "op": "reset",
        "request_id": request_id,
        "episode_id": episode_id,
        "created_ns": time.time_ns(),
    }


def make_success_response(request: dict[str, Any], **payload: Any) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "op": f"{request['op']}_result",
        "request_id": request["request_id"],
        "episode_id": request["episode_id"],
        "status": "ok",
        **payload,
    }


def make_error_response(
    request: dict[str, Any] | None,
    *,
    code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "op": "error",
        "request_id": request.get("request_id") if request else None,
        "episode_id": request.get("episode_id") if request else None,
        "status": "error",
        "error": {"code": code, "message": message},
    }


def validate_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ProtocolError("Server metadata must be a dictionary")
    _validate_version(metadata)
    return metadata


def validate_request(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ProtocolError("Request must be a dictionary")
    _validate_version(message)
    op = message.get("op")
    if op not in SUPPORTED_OPERATIONS:
        raise ProtocolError(f"Unsupported operation: {op!r}")
    if not isinstance(message.get("request_id"), int):
        raise ProtocolError("request_id must be an integer")
    if not isinstance(message.get("episode_id"), str) or not message["episode_id"]:
        raise ProtocolError("episode_id must be a non-empty string")
    if op == "infer":
        if not isinstance(message.get("observation"), dict):
            raise ProtocolError("infer request requires an observation dictionary")
        if not isinstance(message.get("deadline_ms"), int) or message["deadline_ms"] <= 0:
            raise ProtocolError("deadline_ms must be a positive integer")
    return message


def validate_response(message: Any, *, request_id: int) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ProtocolError("Response must be a dictionary")
    _validate_version(message)
    if message.get("request_id") != request_id:
        raise ProtocolError(
            f"Response request_id {message.get('request_id')!r} does not match request {request_id}"
        )
    if message.get("status") not in {"ok", "error"}:
        raise ProtocolError("Response status must be 'ok' or 'error'")
    return message


def validate_inference_result(
    result: Any,
    *,
    allow_missing_goal_reached: bool = False,
) -> dict[str, Any]:
    """Validate the protocol-v2 inference result contract."""
    if not isinstance(result, dict):
        raise ProtocolError("Inference result must be a dictionary")
    if "goal_reached" not in result:
        if allow_missing_goal_reached:
            return result
        raise ProtocolError("Inference result is missing goal_reached")
    if type(result["goal_reached"]) is not bool:
        raise ProtocolError("goal_reached must be a native boolean")
    if not result["goal_reached"] and "actions" not in result:
        raise ProtocolError("Inference result requires actions when goal_reached is false")
    return result


def request_deadline_expired(request: dict[str, Any], *, elapsed_ms: float) -> bool:
    """Check a deadline using elapsed server-local time, avoiding clock skew."""
    deadline_ms = request.get("deadline_ms")
    if not isinstance(deadline_ms, int):
        return True
    return elapsed_ms > deadline_ms


def _validate_version(message: dict[str, Any]) -> None:
    if message.get("protocol") != PROTOCOL_NAME:
        raise ProtocolError(f"Unsupported protocol: {message.get('protocol')!r}")
    if message.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"Unsupported protocol version: {message.get('protocol_version')!r}; expected {PROTOCOL_VERSION}"
        )
