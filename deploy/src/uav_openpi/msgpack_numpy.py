"""MessagePack support for NumPy values, derived from openpi-client.

Modified to add explicit type annotations and input validation. Object,
structured, and complex arrays remain unsupported so serialization never
falls back to pickle.
"""

from __future__ import annotations

import functools
from typing import Any

import msgpack
import numpy as np


def pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    raise TypeError(f"Unsupported value for MessagePack: {type(obj)!r}")


def unpack_array(obj: dict[Any, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array, use_bin_type=True)
packb = functools.partial(msgpack.packb, default=pack_array, use_bin_type=True)
Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array, raw=False)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array, raw=False)
