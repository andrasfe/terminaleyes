"""Camera-to-terminal calibration.

Automatically detects where the terminal display appears in the webcam's
field of view by flashing the display white/black and diffing the frames.
Outputs a crop region that isolates the terminal for accurate MLLM reads.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CalibrationDisplay:
    """Minimal pygame display used only during calibration.

    Flashes between solid white and solid black so the webcam can
    detect which pixels belong to the terminal window.
    """

    def __init__(self, fullscreen: bool = True) -> None:
        self._fullscreen = fullscreen
        self._color: tuple[int, int, int] = (0, 0, 0)
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="calibration-display"
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def set_color(self, color: tuple[int, int, int]) -> None:
        with self._lock:
            self._color = color

    def _loop(self) -> None:
        import pygame
        pygame.init()

        if self._fullscreen:
            info = pygame.display.Info()
            screen = pygame.display.set_mode(
                (info.current_w, info.current_h), pygame.FULLSCREEN
            )
        else:
            screen = pygame.display.set_mode((1024, 768))

        pygame.display.set_caption("terminaleyes - Calibrating...")
        clock = pygame.time.Clock()
        self._ready.set()

        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._running = False
                    break

            with self._lock:
                color = self._color

            screen.fill(color)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


async def calibrate(
    device_index: int = 0,
    fullscreen: bool = True,
    margin: int = 10,
) -> dict:
    """Run the calibration procedure.

    1. Open a display window (fullscreen by default)
    2. Show solid BLACK, capture a webcam frame
    3. Show solid WHITE, capture a webcam frame
    4. Diff the frames to find the changed region
    5. Return the crop coordinates

    Args:
        device_index: Webcam device index.
        fullscreen: Whether the calibration display is fullscreen.
        margin: Pixels of margin to add inside the detected region
                to avoid edge artifacts.

    Returns:
        Dict with keys: crop_x, crop_y, crop_width, crop_height,
        frame_width, frame_height, and a confidence score.
    """
    display = CalibrationDisplay(fullscreen=fullscreen)
    display.start()

    loop = asyncio.get_event_loop()

    # Open webcam
    cap = await loop.run_in_executor(None, cv2.VideoCapture, device_index)
    if not cap.isOpened():
        display.stop()
        raise RuntimeError(f"Cannot open webcam device {device_index}")

    # Let the camera auto-exposure settle with a dark screen
    logger.info("Calibration: warming up camera...")
    display.set_color((0, 0, 0))
    for _ in range(30):
        await loop.run_in_executor(None, cap.read)
        await asyncio.sleep(0.1)

    # Phase 1: BLACK screen -- hold and let exposure settle
    logger.info("Calibration: showing BLACK screen...")
    display.set_color((0, 0, 0))
    await asyncio.sleep(2.5)
    # Flush stale frames
    for _ in range(10):
        await loop.run_in_executor(None, cap.read)

    black_frames = []
    for _ in range(8):
        ret, frame = await loop.run_in_executor(None, cap.read)
        if ret:
            black_frames.append(frame.astype(np.float32))
        await asyncio.sleep(0.1)
    black_avg = np.mean(black_frames, axis=0).astype(np.uint8)

    # Phase 2: WHITE screen -- hold longer for auto-exposure to adapt
    logger.info("Calibration: showing WHITE screen...")
    display.set_color((255, 255, 255))
    await asyncio.sleep(3.0)
    # Flush stale frames
    for _ in range(10):
        await loop.run_in_executor(None, cap.read)

    white_frames = []
    for _ in range(8):
        ret, frame = await loop.run_in_executor(None, cap.read)
        if ret:
            white_frames.append(frame.astype(np.float32))
        await asyncio.sleep(0.1)
    white_avg = np.mean(white_frames, axis=0).astype(np.uint8)

    # Phase 3: Compute difference
    logger.info("Calibration: computing difference...")
    diff = cv2.absdiff(white_avg, black_avg)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Save debug images
    cv2.imwrite("calibration_black.png", black_avg)
    cv2.imwrite("calibration_white.png", white_avg)
    cv2.imwrite("calibration_diff.png", gray_diff)

    # Use adaptive threshold based on actual diff range
    diff_max = gray_diff.max()
    diff_mean = gray_diff.mean()
    logger.info(
        "Calibration diff stats: min=%d, max=%d, mean=%.1f",
        gray_diff.min(), diff_max, diff_mean,
    )

    # Threshold at 30% of max diff or use Otsu for auto
    if diff_max < 10:
        display.stop()
        cap.release()
        raise RuntimeError(
            "Calibration failed: no significant brightness difference detected. "
            "Make sure the camera can see the screen and the room isn't too bright."
        )

    # Normalize diff to full 0-255 range for better thresholding
    normalized = cv2.normalize(gray_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite("calibration_normalized.png", normalized)

    # Use Otsu's method on normalized diff for automatic threshold
    _, thresh = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    cv2.imwrite("calibration_thresh.png", thresh)

    # Find contours and pick the largest
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        display.stop()
        cap.release()
        raise RuntimeError(
            "Calibration failed: could not detect terminal display. "
            "Make sure the camera can see the screen."
        )

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    frame_h, frame_w = black_avg.shape[:2]
    area_ratio = (w * h) / (frame_w * frame_h)

    # Apply margin (shrink inward to avoid edges)
    x = min(x + margin, frame_w - 1)
    y = min(y + margin, frame_h - 1)
    w = max(w - margin * 2, 10)
    h = max(h - margin * 2, 10)

    # Draw the detected region on a debug image
    debug_img = white_avg.copy()
    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imwrite("calibration_result.png", debug_img)

    display.stop()
    cap.release()

    result = {
        "crop_x": x,
        "crop_y": y,
        "crop_width": w,
        "crop_height": h,
        "frame_width": frame_w,
        "frame_height": frame_h,
        "area_ratio": round(area_ratio, 3),
    }

    logger.info(
        "Calibration complete: terminal at (%d,%d) size %dx%d (%.0f%% of frame)",
        x, y, w, h, area_ratio * 100,
    )
    return result


def apply_calibration_to_config(
    config_path: str,
    calibration: dict,
) -> None:
    """Update a YAML config file with calibration results."""
    import yaml
    from pathlib import Path

    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    if "capture" not in data:
        data["capture"] = {}

    data["capture"]["crop_enabled"] = True
    data["capture"]["crop_x"] = calibration["crop_x"]
    data["capture"]["crop_y"] = calibration["crop_y"]
    data["capture"]["crop_width"] = calibration["crop_width"]
    data["capture"]["crop_height"] = calibration["crop_height"]

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info("Saved calibration to %s", path)
