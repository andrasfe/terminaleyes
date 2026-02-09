"""Frame change detection and quality gating.

Provides cheap local checks to avoid unnecessary MLLM API calls
when the screen hasn't changed or the frame quality is poor.
"""

from __future__ import annotations

import cv2
import numpy as np


def has_frame_changed(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    threshold: float = 0.02,
) -> bool:
    """Check if enough pixels changed between two grayscale frames.

    Args:
        prev_gray: Previous frame in grayscale.
        curr_gray: Current frame in grayscale.
        threshold: Fraction of pixels that must differ (0.0-1.0).

    Returns:
        True if the frame changed enough to warrant an MLLM call.
    """
    diff = cv2.absdiff(prev_gray, curr_gray)
    changed = np.count_nonzero(diff > 25) / diff.size
    return changed > threshold


def is_frame_usable(gray: np.ndarray) -> tuple[bool, str]:
    """Check if a grayscale frame is good enough to send to the MLLM.

    Checks for blur (Laplacian variance) and brightness extremes.

    Returns:
        (usable, reason) tuple.
    """
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))
    if blur < 10:
        return False, "too blurry"
    if brightness < 30:
        return False, "too dark (screen off?)"
    if brightness > 245:
        return False, "overexposed"
    return True, "ok"
