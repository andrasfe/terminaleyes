"""Watch loop orchestrator for passive screen observation."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime

import cv2

from terminaleyes.capture.webcam import WebcamCapture
from terminaleyes.watcher.change import has_frame_changed, is_frame_usable
from terminaleyes.watcher.memory import MemoryStore
from terminaleyes.watcher.models import WatchSession
from terminaleyes.watcher.reader import ScreenReader

logger = logging.getLogger(__name__)


class WatchLoop:
    """Orchestrates periodic screen capture, change detection, and MLLM reading."""

    def __init__(
        self,
        capture: WebcamCapture,
        reader: ScreenReader,
        memory: MemoryStore,
        capture_interval_minutes: float = 3.0,
        session_duration_hours: float = 1.0,
        change_threshold: float = 0.02,
    ) -> None:
        self._capture = capture
        self._reader = reader
        self._memory = memory
        self._interval = capture_interval_minutes * 60.0  # seconds
        self._duration = session_duration_hours * 3600.0  # seconds
        self._change_threshold = change_threshold
        self._stopped = False

    def stop(self) -> None:
        """Signal the watch loop to stop."""
        self._stopped = True

    async def run(self, session_id: str | None = None) -> WatchSession:
        """Run the watch loop for the configured duration.

        Args:
            session_id: Optional session identifier. Auto-generated if not provided.

        Returns:
            A WatchSession with all observations and final summary.
        """
        session_id = session_id or uuid.uuid4().hex[:12]
        started_at = datetime.now()
        changes_detected = 0
        prev_gray = None
        frame_counter = 0

        print(f"Watch session {session_id} started at {started_at.strftime('%H:%M:%S')}")
        print(f"Interval: {self._interval:.0f}s, Duration: {self._duration:.0f}s")
        print()

        async with self._capture:
            # First capture: always read + print positioning notes
            first_obs = await self._capture_and_read(frame_counter)
            if first_obs is not None:
                self._memory.add(first_obs)
                frame_counter += 1
                changes_detected += 1
                print(f"[{first_obs.timestamp.strftime('%H:%M:%S')}] "
                      f"Initial read: {first_obs.content_type} "
                      f"({first_obs.application_context or 'unknown'})")
                if first_obs.positioning_notes and first_obs.positioning_notes != "none":
                    print(f"  Positioning: {first_obs.positioning_notes}")
                # Store grayscale for change detection
                frame = await self._capture.capture_frame()
                prev_gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

            elapsed = (datetime.now() - started_at).total_seconds()
            while elapsed < self._duration and not self._stopped:
                remaining = min(self._interval, self._duration - elapsed)
                await asyncio.sleep(remaining)
                if self._stopped:
                    break

                frame = await self._capture.capture_frame()
                curr_gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

                # Quality gate
                usable, reason = is_frame_usable(curr_gray)
                if not usable:
                    print(f"  Skipped frame: {reason}")
                    elapsed = (datetime.now() - started_at).total_seconds()
                    continue

                # Change gate
                if prev_gray is not None and not has_frame_changed(
                    prev_gray, curr_gray, self._change_threshold
                ):
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}] No change detected, skipping MLLM call")
                    elapsed = (datetime.now() - started_at).total_seconds()
                    continue

                prev_gray = curr_gray

                # Read screen via MLLM
                obs = await self._read_frame(frame.image, frame_counter)
                if obs is not None:
                    self._memory.add(obs)
                    frame_counter += 1
                    changes_detected += 1
                    print(f"[{obs.timestamp.strftime('%H:%M:%S')}] "
                          f"{obs.content_type} "
                          f"({obs.application_context or 'unknown'}) "
                          f"confidence={obs.confidence:.2f}")
                    if (obs.positioning_notes
                            and obs.positioning_notes != "none"
                            and obs.confidence < 0.5):
                        print(f"  Positioning: {obs.positioning_notes}")

                elapsed = (datetime.now() - started_at).total_seconds()

        # Generate final summary
        print("\nGenerating session summary...")
        summary = await self._memory.generate_final_summary(
            self._reader._client, self._reader._model
        )

        session = self._memory.to_session(
            session_id=session_id,
            started_at=started_at,
            capture_interval_minutes=self._interval / 60.0,
            changes_detected=changes_detected,
            final_summary=summary,
        )
        return session

    async def _capture_and_read(self, frame_number: int):
        """Capture a frame and read it via MLLM."""
        try:
            frame = await self._capture.capture_frame()
            return await self._read_frame(frame.image, frame_number)
        except Exception as e:
            logger.error("Capture/read failed: %s", e)
            print(f"  Error: {e}")
            return None

    async def _read_frame(self, image, frame_number: int):
        """Read a frame image via MLLM."""
        try:
            return await self._reader.read_screen(image, frame_number)
        except Exception as e:
            logger.error("MLLM read failed: %s", e)
            print(f"  MLLM error: {e}")
            return None
