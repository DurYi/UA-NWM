"""Client-side image utilities derived from openpi_client.image_tools."""

from __future__ import annotations

import numpy as np
from PIL import Image


def convert_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float images to compact uint8 representation."""
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def resize_long_edge(
    image: np.ndarray,
    max_edge: int,
    method: int = Image.Resampling.LANCZOS,
) -> np.ndarray:
    """Downsample one uint8 HWC image while preserving its aspect ratio."""
    image = np.asarray(image)
    if max_edge < 1:
        raise ValueError("max_edge must be positive")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError("image must be a uint8 HWC array with 1, 3, or 4 channels")
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_edge:
        return np.ascontiguousarray(image)
    scale = float(max_edge) / float(longest)
    resized_hw = (max(1, round(width * scale)), max(1, round(height * scale)))
    pil_input = image[..., 0] if image.shape[-1] == 1 else image
    resized = np.asarray(Image.fromarray(pil_input).resize(resized_hw, resample=method), dtype=np.uint8)
    if image.shape[-1] == 1:
        resized = resized[..., None]
    return np.ascontiguousarray(resized)


def resize_with_pad(images: np.ndarray, height: int, width: int, method: int = Image.Resampling.BILINEAR) -> np.ndarray:
    """Resize HWC images without distortion and pad remaining pixels with zero."""
    if images.shape[-3:-1] == (height, width):
        return images
    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    current_width, current_height = image.size
    ratio = max(current_width / width, current_height / height)
    resized_height = max(1, int(current_height / ratio))
    resized_width = max(1, int(current_width / ratio))
    resized_image = image.resize((resized_width, resized_height), resample=method)
    padded = Image.new(resized_image.mode, (width, height), 0)
    padded.paste(resized_image, ((width - resized_width) // 2, (height - resized_height) // 2))
    return padded
