"""Predictive Prefetch: detect confidence decline trends and trigger preemptive cloud upload.

Phase 4 optimization — when confidence is trending down, start uploading context
to cloud before the formal cascade request, reducing effective latency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PrefetchState:
    """Tracks prefetch decisions and their outcomes."""

    prefetch_triggered: int = 0
    prefetch_useful: int = 0  # followed by actual cascade within N frames
    prefetch_wasted: int = 0  # no cascade happened (wasted bandwidth)


class PredictivePrefetcher:
    """Detect confidence trend decline and trigger preemptive cloud upload.

    Monitors a sliding window of confidence values. When the linear trend
    slope falls below a threshold, signals that cloud context should be
    pre-uploaded so the cloud is "warm" when cascade actually fires.
    """

    def __init__(
        self,
        window_size: int = 5,
        decline_threshold: float = -0.05,
        lookahead_frames: int = 10,
    ):
        self.window_size = window_size
        self.decline_threshold = decline_threshold
        self.lookahead_frames = lookahead_frames
        self.confidence_history: deque[float] = deque(maxlen=200)
        self.state = PrefetchState()
        self._pending_prefetch_frame: int | None = None
        self._frame_counter: int = 0

    def update(self, confidence: float) -> None:
        """Record a new confidence observation."""
        self.confidence_history.append(confidence)
        self._frame_counter += 1

        # Check if a pending prefetch was useful (cascade happened within lookahead)
        if self._pending_prefetch_frame is not None:
            if self._frame_counter - self._pending_prefetch_frame > self.lookahead_frames:
                self.state.prefetch_wasted += 1
                self._pending_prefetch_frame = None

    def mark_cascade_happened(self) -> None:
        """Call when a cascade actually fires — validates pending prefetch."""
        if self._pending_prefetch_frame is not None:
            self.state.prefetch_useful += 1
            self._pending_prefetch_frame = None

    def should_prefetch(self) -> bool:
        """Check if cloud context should be pre-uploaded.

        Returns True when confidence shows a declining trend.
        """
        if len(self.confidence_history) < self.window_size:
            return False

        recent = list(self.confidence_history)[-self.window_size:]
        trend = self.get_trend()
        if trend is not None and trend < self.decline_threshold:
            self.state.prefetch_triggered += 1
            self._pending_prefetch_frame = self._frame_counter
            return True
        return False

    def get_trend(self) -> float | None:
        """Compute linear trend of recent confidence values.

        Returns slope (negative = declining), or None if not enough data.
        """
        if len(self.confidence_history) < self.window_size:
            return None
        recent = list(self.confidence_history)[-self.window_size:]
        x = np.arange(self.window_size, dtype=float)
        coeffs = np.polyfit(x, recent, 1)
        return float(coeffs[0])

    def get_stats(self) -> dict:
        return {
            "prefetch_triggered": self.state.prefetch_triggered,
            "prefetch_useful": self.state.prefetch_useful,
            "prefetch_wasted": self.state.prefetch_wasted,
            "precision": round(
                self.state.prefetch_useful / max(1, self.state.prefetch_triggered), 3
            ),
            "history_length": len(self.confidence_history),
        }
