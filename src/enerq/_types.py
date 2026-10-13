"""Shared type aliases."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
FloatLike: TypeAlias = float | ArrayLike

__all__ = ["ArrayLike", "FloatArray", "FloatLike"]
