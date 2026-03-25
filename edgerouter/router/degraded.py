"""Degraded Mode: network disconnection fallback to edge-only with conservative thresholds.

Phase 4 optimization — when WAN is unavailable or latency exceeds a threshold,
automatically fall back to edge-only processing with more conservative thresholds
(prefer false alarms over missed anomalies). Cache data for batch upload when
network recovers.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DegradedModeState:
    """Tracks degraded mode activations and cache state."""

    activations: int = 0
    total_degraded_frames: int = 0
    cache_flushes: int = 0
    total_cached_items: int = 0
    current_activation_start: float | None = None


class DegradedModeController:
    """Handle network disconnection: fall back to edge-only with conservative thresholds.

    Industrial sites have unreliable networks. When cloud is unreachable:
    1. All analysis falls back to edge
    2. Confidence threshold is raised (more conservative = fewer missed anomalies)
    3. Pending data is cached locally for batch upload when network recovers
    """

    def __init__(
        self,
        conservative_threshold_boost: float = 0.15,
        max_cache_size: int = 1000,
        wan_timeout_ms: float = 2000.0,
    ):
        self.conservative_boost = conservative_threshold_boost
        self.max_cache_size = max_cache_size
        self.wan_timeout_ms = wan_timeout_ms
        self.is_degraded: bool = False
        self._cache: deque[dict[str, Any]] = deque(maxlen=max_cache_size)
        self.state = DegradedModeState()

    def enter_degraded_mode(self) -> None:
        """Activate degraded mode (network down)."""
        if not self.is_degraded:
            self.is_degraded = True
            self.state.activations += 1
            self.state.current_activation_start = time.time()

    def exit_degraded_mode(self) -> None:
        """Deactivate degraded mode (network recovered)."""
        if self.is_degraded:
            self.is_degraded = False
            self.state.current_activation_start = None

    def tick(self) -> None:
        """Call once per frame to track degraded frame count."""
        if self.is_degraded:
            self.state.total_degraded_frames += 1

    def get_effective_threshold(self, base_threshold: float) -> float:
        """Return a more conservative threshold when in degraded mode.

        Higher threshold → edge must be more confident to avoid escalation,
        but since cloud is unavailable, this means more cautious actions.
        In practice: edge uses conservative (warning-biased) judgments.
        """
        if self.is_degraded:
            return min(0.95, base_threshold + self.conservative_boost)
        return base_threshold

    def should_force_edge(self) -> bool:
        """In degraded mode, all routing should stay on edge."""
        return self.is_degraded

    def cache_for_upload(self, data: dict[str, Any]) -> bool:
        """Cache data for batch upload when network recovers.

        Returns True if cached successfully, False if cache is full.
        """
        if len(self._cache) >= self.max_cache_size:
            return False
        self._cache.append(data)
        self.state.total_cached_items += 1
        return True

    def flush_cache(self) -> list[dict[str, Any]]:
        """Retrieve and clear all cached data (call when network recovers)."""
        items = list(self._cache)
        self._cache.clear()
        self.state.cache_flushes += 1
        return items

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def get_stats(self) -> dict:
        return {
            "is_degraded": self.is_degraded,
            "activations": self.state.activations,
            "total_degraded_frames": self.state.total_degraded_frames,
            "cache_size": self.cache_size,
            "total_cached_items": self.state.total_cached_items,
            "cache_flushes": self.state.cache_flushes,
        }
