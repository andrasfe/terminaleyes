"""Image processing utilities for terminaleyes.

Shared image encoding, conversion, and preprocessing functions used
by the capture and interpreter modules.
"""

from __future__ import annotations

import base64
import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def numpy_to_base64_png(image: np.ndarray) -> str:
    """Convert a numpy image array (BGR, OpenCV format) to base64 PNG."""
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """Convert a numpy image array (BGR) to a PIL Image (RGB)."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) to a numpy array (BGR)."""
    rgb_array = np.array(image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def resize_for_mllm(
    image: np.ndarray,
    max_dimension: int = 1568,
) -> np.ndarray:
    """Resize an image to fit within MLLM input size limits.

    Preserves aspect ratio. Returns the original if already within limits.
    """
    h, w = image.shape[:2]
    if h <= max_dimension and w <= max_dimension:
        return image
    scale = max_dimension / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
