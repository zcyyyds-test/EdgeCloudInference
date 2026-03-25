"""Tests for Phase 4 features: Predictive Prefetch + Degraded Mode."""

import pytest

from edgerouter.router.prefetch import PredictivePrefetcher
from edgerouter.router.degraded import DegradedModeController


# -----------------------------------------------------------------------
# Predictive Prefetch
# -----------------------------------------------------------------------

class TestPredictivePrefetcher:
    def test_no_prefetch_with_insufficient_data(self):
        pf = PredictivePrefetcher(window_size=5)
        pf.update(0.9)
        pf.update(0.8)
        assert not pf.should_prefetch()

    def test_no_prefetch_on_stable_confidence(self):
        pf = PredictivePrefetcher(window_size=5, decline_threshold=-0.05)
        for _ in range(10):
            pf.update(0.85)
        assert not pf.should_prefetch()

    def test_prefetch_on_declining_confidence(self):
        pf = PredictivePrefetcher(window_size=5, decline_threshold=-0.03)
        # Inject a clear decline: 0.9, 0.8, 0.7, 0.6, 0.5
        for v in [0.9, 0.8, 0.7, 0.6, 0.5]:
            pf.update(v)
        assert pf.should_prefetch()

    def test_get_trend_returns_negative_on_decline(self):
        pf = PredictivePrefetcher(window_size=5)
        for v in [0.9, 0.85, 0.8, 0.75, 0.7]:
            pf.update(v)
        trend = pf.get_trend()
        assert trend is not None
        assert trend < 0

    def test_get_trend_returns_none_insufficient_data(self):
        pf = PredictivePrefetcher(window_size=5)
        pf.update(0.5)
        assert pf.get_trend() is None

    def test_mark_cascade_validates_prefetch(self):
        pf = PredictivePrefetcher(window_size=3, decline_threshold=-0.01)
        for v in [0.9, 0.7, 0.5]:
            pf.update(v)
        pf.should_prefetch()  # triggers prefetch
        pf.mark_cascade_happened()
        stats = pf.get_stats()
        assert stats["prefetch_useful"] >= 1

    def test_wasted_prefetch_tracked(self):
        pf = PredictivePrefetcher(window_size=3, decline_threshold=-0.01, lookahead_frames=2)
        for v in [0.9, 0.7, 0.5]:
            pf.update(v)
        pf.should_prefetch()  # triggers prefetch
        # No cascade happens, advance frames beyond lookahead
        for _ in range(5):
            pf.update(0.9)  # stable again, each update increments frame counter
        stats = pf.get_stats()
        assert stats["prefetch_wasted"] >= 1

    def test_stats_structure(self):
        pf = PredictivePrefetcher()
        stats = pf.get_stats()
        assert "prefetch_triggered" in stats
        assert "prefetch_useful" in stats
        assert "prefetch_wasted" in stats
        assert "precision" in stats


# -----------------------------------------------------------------------
# Degraded Mode
# -----------------------------------------------------------------------

class TestDegradedModeController:
    def test_initial_state_not_degraded(self):
        dm = DegradedModeController()
        assert not dm.is_degraded
        assert not dm.should_force_edge()

    def test_enter_degraded_mode(self):
        dm = DegradedModeController()
        dm.enter_degraded_mode()
        assert dm.is_degraded
        assert dm.should_force_edge()

    def test_exit_degraded_mode(self):
        dm = DegradedModeController()
        dm.enter_degraded_mode()
        dm.exit_degraded_mode()
        assert not dm.is_degraded

    def test_conservative_threshold_in_degraded(self):
        dm = DegradedModeController(conservative_threshold_boost=0.15)
        base = 0.7
        assert dm.get_effective_threshold(base) == 0.7  # not degraded
        dm.enter_degraded_mode()
        assert dm.get_effective_threshold(base) == 0.85  # boosted

    def test_threshold_capped_at_095(self):
        dm = DegradedModeController(conservative_threshold_boost=0.5)
        dm.enter_degraded_mode()
        assert dm.get_effective_threshold(0.9) == 0.95

    def test_cache_and_flush(self):
        dm = DegradedModeController(max_cache_size=5)
        dm.enter_degraded_mode()
        for i in range(3):
            assert dm.cache_for_upload({"frame": i})
        assert dm.cache_size == 3

        items = dm.flush_cache()
        assert len(items) == 3
        assert dm.cache_size == 0

    def test_cache_overflow(self):
        dm = DegradedModeController(max_cache_size=2)
        dm.cache_for_upload({"a": 1})
        dm.cache_for_upload({"a": 2})
        assert not dm.cache_for_upload({"a": 3})  # full
        assert dm.cache_size == 2

    def test_tick_tracks_degraded_frames(self):
        dm = DegradedModeController()
        dm.tick()  # not degraded
        dm.enter_degraded_mode()
        dm.tick()
        dm.tick()
        stats = dm.get_stats()
        assert stats["total_degraded_frames"] == 2

    def test_activation_count(self):
        dm = DegradedModeController()
        dm.enter_degraded_mode()
        dm.exit_degraded_mode()
        dm.enter_degraded_mode()
        dm.exit_degraded_mode()
        assert dm.state.activations == 2

    def test_stats_structure(self):
        dm = DegradedModeController()
        stats = dm.get_stats()
        assert "is_degraded" in stats
        assert "activations" in stats
        assert "cache_size" in stats
