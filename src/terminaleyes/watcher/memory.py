"""Observation storage and session summary generation."""

from __future__ import annotations

import logging
from datetime import datetime

from terminaleyes.watcher.models import ScreenObservation, WatchSession

logger = logging.getLogger(__name__)


class MemoryStore:
    """Accumulates screen observations and generates a final summary."""

    def __init__(self) -> None:
        self._observations: list[ScreenObservation] = []

    @property
    def observations(self) -> list[ScreenObservation]:
        return list(self._observations)

    @property
    def count(self) -> int:
        return len(self._observations)

    def add(self, obs: ScreenObservation) -> None:
        self._observations.append(obs)

    async def generate_final_summary(self, client, model: str) -> str:
        """Generate a summary of all observations with a single MLLM call.

        Args:
            client: An AsyncOpenAI client instance.
            model: Model name to use for the summary.

        Returns:
            Summary text describing what was observed during the session.
        """
        if not self._observations:
            return "No observations recorded."

        obs_texts = []
        for obs in self._observations:
            ts = obs.timestamp.strftime("%H:%M:%S")
            text_preview = obs.visible_text[:300]
            obs_texts.append(
                f"[{ts}] {obs.content_type}"
                f" ({obs.application_context or 'unknown app'})"
                f" - {text_preview}"
            )

        observations_block = "\n".join(obs_texts)
        first_ts = self._observations[0].timestamp.strftime("%H:%M")
        last_ts = self._observations[-1].timestamp.strftime("%H:%M")

        prompt = (
            f"Here are {len(self._observations)} screen observations"
            f" from {first_ts} to {last_ts}.\n\n"
            f"{observations_block}\n\n"
            "Summarize what the user was doing during this session."
            " Be concise and factual — only describe what was observed."
        )

        try:
            response = await client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Failed to generate summary: %s", e)
            return f"Summary generation failed: {e}"

    def to_session(
        self,
        session_id: str,
        started_at: datetime,
        capture_interval_minutes: float,
        changes_detected: int,
        final_summary: str = "",
    ) -> WatchSession:
        ended_at = datetime.now()
        duration = (ended_at - started_at).total_seconds() / 60.0
        return WatchSession(
            session_id=session_id,
            started_at=started_at,
            ended_at=ended_at,
            duration_minutes=round(duration, 2),
            capture_interval_minutes=capture_interval_minutes,
            total_captures=len(self._observations),
            changes_detected=changes_detected,
            observations=list(self._observations),
            final_summary=final_summary,
        )
