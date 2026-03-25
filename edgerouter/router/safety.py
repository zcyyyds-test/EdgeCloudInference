"""Safety classifier: detect critical conditions requiring immediate edge control."""

from __future__ import annotations

from edgerouter.core.config import RouterConfig
from edgerouter.core.schema import VisionOutput


class SafetyClassifier:
    """Hard-coded safety rules — no ML, pure threshold logic.

    Critical conditions bypass the router entirely and trigger
    immediate edge control actions.
    """

    def __init__(self, config: RouterConfig | None = None):
        cfg = config or RouterConfig()
        self.critical_upper = cfg.critical_anomaly_upper
        self.critical_lower = cfg.critical_anomaly_lower
        self.critical_rate = cfg.critical_anomaly_rate

    def is_critical(
        self,
        vision_output: VisionOutput,
        prev_output: VisionOutput | None = None,
    ) -> bool:
        """Return True if the situation requires immediate emergency action."""
        # Rule 1: anomaly level near upper critical threshold
        if vision_output.anomaly_level >= self.critical_upper:
            return True

        # Rule 2: anomaly level near lower critical threshold
        if vision_output.anomaly_level <= self.critical_lower:
            return True

        # Rule 3: sudden level change (possible failure or surge)
        if prev_output is not None:
            dt = vision_output.timestamp - prev_output.timestamp
            if dt > 0:
                rate = abs(vision_output.anomaly_level - prev_output.anomaly_level) / dt
                if rate >= self.critical_rate:
                    return True

        return False

    def get_reason(
        self,
        vision_output: VisionOutput,
        prev_output: VisionOutput | None = None,
    ) -> str:
        """Return a human-readable reason for the critical classification."""
        if vision_output.anomaly_level >= self.critical_upper:
            return f"upper_bound_breach: level={vision_output.anomaly_level:.1f} >= {self.critical_upper}"
        if vision_output.anomaly_level <= self.critical_lower:
            return f"lower_bound_breach: level={vision_output.anomaly_level:.1f} <= {self.critical_lower}"
        if prev_output is not None:
            dt = vision_output.timestamp - prev_output.timestamp
            if dt > 0:
                rate = abs(vision_output.anomaly_level - prev_output.anomaly_level) / dt
                if rate >= self.critical_rate:
                    return f"sudden_change: rate={rate:.1f}/s >= {self.critical_rate}"
        return "unknown"
