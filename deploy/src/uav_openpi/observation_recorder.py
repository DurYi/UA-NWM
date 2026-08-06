"""Asynchronous server-side persistence for accepted client observations."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)
_STOP = object()


@dataclass(frozen=True)
class _ObservationRecord:
    episode_id: str
    request_id: int
    received_ns: int
    images: dict[str, np.ndarray]
    state: np.ndarray
    fields: dict[str, Any]


@dataclass(frozen=True)
class _InferenceRecord:
    episode_id: str
    request_id: int
    received_ns: int
    completed_ns: int
    result: dict[str, Any]
    server_timing: dict[str, float]


class ObservationRecorder:
    """Copy observations into a bounded queue and persist them off the inference path."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        image_format: str = "png",
        jpeg_quality: int = 95,
        queue_size: int = 64,
    ) -> None:
        normalized_format = image_format.lower()
        if normalized_format == "jpg":
            normalized_format = "jpeg"
        if normalized_format not in {"png", "jpeg"}:
            raise ValueError("image_format must be 'png' or 'jpeg'")
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if int(queue_size) < 1:
            raise ValueError("queue_size must be positive")

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_format = normalized_format
        self.jpeg_quality = int(jpeg_quality)
        # A request contributes one observation item and one inference item.
        self._queue: queue.Queue[_ObservationRecord | _InferenceRecord | object] = queue.Queue(
            maxsize=int(queue_size) * 2
        )
        self._closed = False
        self._close_lock = threading.Lock()
        self._dropped = 0
        self._saved = 0
        self._saved_inferences = 0
        self._worker = threading.Thread(
            target=self._run,
            name="uav-observation-recorder",
            daemon=True,
        )
        self._worker.start()
        logger.info(
            "Observation recording enabled dir=%s image_format=%s queue_size=%s",
            self.output_dir,
            self.image_format,
            queue_size,
        )

    @property
    def saved_count(self) -> int:
        return self._saved

    @property
    def dropped_count(self) -> int:
        return self._dropped

    @property
    def saved_inference_count(self) -> int:
        return self._saved_inferences

    def record(
        self,
        observation: Mapping[str, Any],
        *,
        request_id: int,
        episode_id: str,
        received_ns: int | None = None,
    ) -> bool:
        """Queue one immutable snapshot; return false rather than blocking when full."""
        images_value = observation.get("images", {})
        if not isinstance(images_value, Mapping):
            raise ValueError("observation images must be a mapping")
        images = {str(name): np.ascontiguousarray(np.asarray(image).copy()) for name, image in images_value.items()}
        state = np.ascontiguousarray(np.asarray(observation.get("state")).copy())
        fields = {
            str(key): _to_jsonable(value)
            for key, value in observation.items()
            if key not in {"images", "state"}
        }
        item = _ObservationRecord(
            episode_id=str(episode_id),
            request_id=int(request_id),
            received_ns=time.time_ns() if received_ns is None else int(received_ns),
            images=images,
            state=state,
            fields=fields,
        )
        with self._close_lock:
            if self._closed:
                logger.warning("Observation recorder is closed; dropping request_id=%s", request_id)
                return False
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self._dropped += 1
                logger.warning(
                    "Observation recording queue full; dropping episode_id=%s request_id=%s dropped=%s",
                    episode_id,
                    request_id,
                    self._dropped,
                )
                return False
        return True

    def record_inference(
        self,
        result: Mapping[str, Any],
        *,
        request_id: int,
        episode_id: str,
        received_ns: int,
        completed_ns: int | None = None,
        queue_ms: float,
        infer_ms: float,
    ) -> bool:
        """Queue the action, policy debug, and server timing for one recorded observation."""
        item = _InferenceRecord(
            episode_id=str(episode_id),
            request_id=int(request_id),
            received_ns=int(received_ns),
            completed_ns=time.time_ns() if completed_ns is None else int(completed_ns),
            result={str(key): _to_jsonable(value) for key, value in result.items()},
            server_timing={"queue_ms": float(queue_ms), "infer_ms": float(infer_ms)},
        )
        with self._close_lock:
            if self._closed:
                logger.warning("Observation recorder is closed; dropping inference request_id=%s", request_id)
                return False
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self._dropped += 1
                logger.warning(
                    "Inference recording queue full; dropping episode_id=%s request_id=%s dropped=%s",
                    episode_id,
                    request_id,
                    self._dropped,
                )
                return False
        return True

    def close(self) -> None:
        """Flush queued observations and stop the writer thread."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(_STOP)
        self._worker.join()
        logger.info(
            "Observation recorder stopped dir=%s observations=%s inferences=%s dropped=%s",
            self.output_dir,
            self._saved,
            self._saved_inferences,
            self._dropped,
        )

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                if isinstance(item, _ObservationRecord):
                    self._write_observation(item)
                    self._saved += 1
                else:
                    assert isinstance(item, _InferenceRecord)
                    self._write_inference(item)
                    self._saved_inferences += 1
            except Exception:
                logger.exception("Failed to persist an observation")
            finally:
                self._queue.task_done()

    def _write_observation(self, item: _ObservationRecord) -> None:
        episode_name = _safe_name(item.episode_id, fallback="episode")
        episode_dir = self.output_dir / episode_name
        record_name = _record_name(item.request_id, item.received_ns)
        record_dir = episode_dir / record_name
        record_dir.mkdir(parents=True, exist_ok=False)

        image_files: dict[str, str] = {}
        used_names: set[str] = set()
        for index, (image_key, image) in enumerate(item.images.items()):
            base_name = _safe_name(image_key, fallback=f"image_{index:02d}")
            if base_name in used_names:
                base_name = f"{base_name}_{index:02d}"
            used_names.add(base_name)
            suffix = ".png" if self.image_format == "png" else ".jpg"
            image_path = record_dir / f"{base_name}{suffix}"
            self._save_image(image_path, image)
            image_files[image_key] = image_path.name

        state_path = record_dir / "state.npy"
        np.save(state_path, item.state, allow_pickle=False)
        metadata = {
            "schema_version": 1,
            "episode_id": item.episode_id,
            "request_id": item.request_id,
            "received_ns": item.received_ns,
            "image_files": image_files,
            "state_file": state_path.name,
            "state_dtype": str(item.state.dtype),
            "state_shape": list(item.state.shape),
            "fields": item.fields,
        }
        metadata_path = record_dir / "observation.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

        episode_dir.mkdir(parents=True, exist_ok=True)
        manifest_entry = {**metadata, "record_dir": record_name}
        with (episode_dir / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest_entry, ensure_ascii=True, separators=(",", ":")) + "\n")
        logger.info(
            "Recorded observation episode_id=%s request_id=%s images=%s dir=%s",
            item.episode_id,
            item.request_id,
            list(item.images),
            record_dir,
        )

    def _write_inference(self, item: _InferenceRecord) -> None:
        episode_name = _safe_name(item.episode_id, fallback="episode")
        episode_dir = self.output_dir / episode_name
        record_name = _record_name(item.request_id, item.received_ns)
        record_dir = episode_dir / record_name
        if not record_dir.is_dir():
            raise FileNotFoundError(f"observation record directory is missing: {record_dir}")

        result = item.result
        metadata = {
            "schema_version": 1,
            "episode_id": item.episode_id,
            "request_id": item.request_id,
            "observation_received_ns": item.received_ns,
            "inference_completed_ns": item.completed_ns,
            "goal_reached": bool(result["goal_reached"]),
            "actions": result.get("actions"),
            "debug": result.get("debug"),
            "server_timing": item.server_timing,
        }
        inference_path = record_dir / "inference.json"
        inference_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        manifest_entry = {
            "schema_version": 1,
            "episode_id": item.episode_id,
            "request_id": item.request_id,
            "observation_received_ns": item.received_ns,
            "inference_completed_ns": item.completed_ns,
            "goal_reached": metadata["goal_reached"],
            "actions": metadata["actions"],
            "server_timing": item.server_timing,
            "record_dir": record_name,
            "inference_file": inference_path.name,
        }
        with (episode_dir / "inference_index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest_entry, ensure_ascii=True, separators=(",", ":")) + "\n")
        logger.info(
            "Recorded inference episode_id=%s request_id=%s goal_reached=%s actions=%s file=%s",
            item.episode_id,
            item.request_id,
            metadata["goal_reached"],
            metadata["actions"],
            inference_path,
        )

    def _save_image(self, path: Path, image: np.ndarray) -> None:
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
            raise ValueError("recorded images must be uint8 HWC arrays with 1, 3, or 4 channels")
        pil_input = array[..., 0] if array.shape[-1] == 1 else array
        pil_image = Image.fromarray(pil_input)
        if self.image_format == "png":
            pil_image.save(path, format="PNG", compress_level=3)
            return
        if pil_image.mode not in {"L", "RGB"}:
            pil_image = pil_image.convert("RGB")
        pil_image.save(path, format="JPEG", quality=self.jpeg_quality, subsampling=0)


def _safe_name(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return normalized[:120] or fallback


def _record_name(request_id: int, received_ns: int) -> str:
    return f"request_{int(request_id):08d}_{int(received_ns)}"


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return repr(value)
