"""HSV-based cursor finder.

Premise: classical computer vision can locate the mouse cursor in a
webcam frame in milliseconds — but only if the cursor itself stands out.

**Ubuntu / GNOME setup (one-time, on the target machine):**

    sudo apt install -y xcursor-themes
    gsettings set org.gnome.desktop.interface cursor-theme 'redglass'
    gsettings set org.gnome.desktop.interface cursor-size 96
    # log out / log in (or open a fresh app) for the change to take effect

``redglass`` is a saturated-red X11 cursor that ships with the
``xcursor-themes`` package on Ubuntu. At size 96 it's roughly 50 px on
a webcam viewing a 1080p screen — easy to find by HSV thresholding for
red. No models, no diff, no per-step token spend.

If detection misses (cursor not set up, or unusual webcam colour cast),
this returns ``None`` and ``setup_instructions()`` prints what to do.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# Red wraps in HSV; cover both ends. Saturation/value floors keep us
# from latching onto pinkish skin tones, brown text, etc.
RED_LO_A = np.array([0,   140, 110], dtype=np.uint8)
RED_HI_A = np.array([10,  255, 255], dtype=np.uint8)
RED_LO_B = np.array([170, 140, 110], dtype=np.uint8)
RED_HI_B = np.array([180, 255, 255], dtype=np.uint8)


# Spatial constraints (image-percent / image-area).
# At cursor-size 96 on a 1080p screen filling ~70% of a 1280×720 webcam,
# the cursor occupies ~30–60 px → ~900–3600 px². Allow a wide range.
MIN_AREA_PCT = 0.00015   # ~140 px on a 1280×720 frame
MAX_AREA_PCT = 0.020     # ~18 000 px on the same frame


@dataclass
class CursorHit:
    x_pct: float
    y_pct: float
    area_pct: float
    confidence: float  # 0..1, blends size + colour purity


def find_cursor_hsv(image_bgr: np.ndarray) -> CursorHit | None:
    """Find the saturated-red ``redglass`` cursor in a BGR frame.

    Returns ``None`` if no plausible candidate is present.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        return None
    h, w = image_bgr.shape[:2]
    img_area = h * w
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    red_a = cv2.inRange(hsv, RED_LO_A, RED_HI_A)
    red_b = cv2.inRange(hsv, RED_LO_B, RED_HI_B)
    red_mask = cv2.bitwise_or(red_a, red_b)

    # Clean small noise; close so the cursor's outline+body merge into
    # one blob even if the centre highlight breaks the red region.
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel3)
    kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel5)

    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    best: CursorHit | None = None
    for c in contours:
        area = cv2.contourArea(c)
        area_pct = area / img_area
        if area_pct < MIN_AREA_PCT:
            continue
        if area_pct > MAX_AREA_PCT:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        # Confidence: prefer mid-range blob sizes (cursor-sized).
        size_score = min(1.0, area_pct / 0.005)
        # Penalize blobs at extreme image edges (bezel/environment).
        edge_score = 1.0
        margin = 0.02
        if (cx / w < margin or cx / w > 1 - margin
                or cy / h < margin or cy / h > 1 - margin):
            edge_score = 0.3
        confidence = size_score * edge_score
        if best is None or confidence > best.confidence:
            best = CursorHit(
                x_pct=cx / w, y_pct=cy / h,
                area_pct=area_pct, confidence=confidence,
            )
    return best


def annotate_cursor(
    image_bgr: np.ndarray, hit: CursorHit,
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """Return a copy of ``image_bgr`` with the cursor hit marked."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    cx = int(hit.x_pct * w)
    cy = int(hit.y_pct * h)
    cv2.circle(out, (cx, cy), 22, color, 2)
    cv2.line(out, (cx - 30, cy), (cx + 30, cy), color, 1)
    cv2.line(out, (cx, cy - 30), (cx, cy + 30), color, 1)
    cv2.putText(
        out, f"cursor a={hit.area_pct:.4f} c={hit.confidence:.2f}",
        (cx + 24, cy - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
    )
    return out


def find_cursor_by_variance(
    frames: list[np.ndarray],
    variance_threshold: float = 8.0,
    min_active_pixels: int = 30,
    max_active_fraction: float = 0.05,
) -> tuple[float, float] | None:
    """Find the cursor by computing the centroid of all high-variance
    pixels across frames captured during cursor oscillation.

    Premise: when we jiggle the cursor (send small ±moves and back),
    only the cursor's pixels change between frames. Static UI ≈ 0
    variance. The cursor leaves a small "trail" of high-variance
    pixels at each oscillation position. The centroid of that trail
    is approximately the cursor's start position (oscillation is
    symmetric, so the trajectory centres on where it started).

    Uses a fixed variance threshold (default 8) rather than a
    percentile — with ~hundreds of moving pixels in a multi-million
    pixel frame, the 99th percentile is 0 and the percentile-based
    cutoff would short-circuit. Cursor pixels have std ≈ 40–80
    across the oscillation, well above 8.

    Returns ``None`` if the variance signal is too weak
    (``< min_active_pixels``) or too broad (more than
    ``max_active_fraction`` of the image is "active" — likely
    something other than the cursor is animating).
    """
    if len(frames) < 3:
        return None
    if any(f.ndim != 2 for f in frames):
        frames = [
            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            for f in frames
        ]
    h, w = frames[0].shape[:2]
    img_area = h * w
    arr = np.stack([f.astype(np.float32) for f in frames], axis=0)
    var = arr.std(axis=0)
    mask = (var > variance_threshold).astype(np.uint8) * 255
    # Open to drop isolated noise pixels; keep cursor outlines intact.
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)

    active_pixels = int((mask > 0).sum())
    if active_pixels < min_active_pixels:
        return None
    if active_pixels > img_area * max_active_fraction:
        # Too many pixels are moving — something else is animating.
        return None

    # Centroid of the entire active mask. Robust to multiple
    # disconnected blobs (cursor visited several positions during the
    # jiggle): the geometric centre of all positions ≈ the original
    # cursor position when the oscillation pattern is symmetric.
    M = cv2.moments(mask, binaryImage=True)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return cx / w, cy / h


def setup_instructions() -> str:
    """One-shot setup for the target Ubuntu machine's cursor.

    Print to the user when ``find_cursor_hsv`` keeps returning
    ``None`` so they can switch to the bright-red redglass theme.
    """
    return (
        "═══════════ CURSOR SETUP REQUIRED (Ubuntu) ═══════════\n"
        "On the TARGET machine, run:\n"
        "  sudo apt install -y xcursor-themes\n"
        "  gsettings set org.gnome.desktop.interface cursor-theme 'redglass'\n"
        "  gsettings set org.gnome.desktop.interface cursor-size 96\n"
        "Then log out and back in (or open a new app) so the cursor\n"
        "actually changes. Why: the homer locates the cursor by HSV\n"
        "thresholding for saturated red — redglass at size 96 is\n"
        "unmistakable on a webcam, no model calls needed.\n"
        "══════════════════════════════════════════════════════"
    )
