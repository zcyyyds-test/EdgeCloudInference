"""Control engine: edge-resident, executes control actions based on analysis."""

from __future__ import annotations

import logging

from edgerouter.core.schema import (
    AnalysisResult,
    ControlAction,
    Judgment,
    RoutingDecision,
    RoutingTier,
)

logger = logging.getLogger(__name__)


class ControlEngine:
    """Edge control engine.

    Always resident on edge. Receives analysis results and routing
    decisions, outputs control actions. Supports cloud revision for
    cascade scenarios.
    """

    def execute(
        self,
        analysis: AnalysisResult,
        routing_decision: RoutingDecision,
    ) -> ControlAction:
        """Determine control action from analysis result."""

        # Emergency: immediate safety action, don't wait for analysis
        if routing_decision.tier == RoutingTier.EDGE_EMERGENCY:
            return self._emergency_action(analysis)

        if analysis.judgment == Judgment.NORMAL:
            return ControlAction(
                type="maintain",
                based_on_judgment=Judgment.NORMAL,
            )

        if analysis.judgment == Judgment.WARNING:
            action = self._conservative_action(analysis)
            # In cascade mode, mark as pending cloud revision
            if routing_decision.tier == RoutingTier.CASCADE:
                action.pending_cloud_revision = True
            return action

        if analysis.judgment == Judgment.ALARM:
            return self._alarm_action(analysis)

        # Fallback
        return ControlAction(type="maintain", based_on_judgment=analysis.judgment)

    def revise_action(
        self,
        current_action: ControlAction,
        cloud_analysis: AnalysisResult,
    ) -> ControlAction:
        """Revise a pending action after cloud analysis returns.

        If cloud confirms edge → keep current action.
        If cloud disagrees → update to cloud's recommendation.
        """
        if not current_action.pending_cloud_revision:
            return current_action

        if cloud_analysis.judgment == current_action.based_on_judgment:
            # Cloud confirms edge — clear pending flag
            current_action.pending_cloud_revision = False
            logger.info("Cloud confirmed edge judgment: %s", cloud_analysis.judgment.value)
            return current_action

        # Cloud disagrees — apply cloud's recommendation
        logger.info(
            "Cloud revised edge judgment: %s → %s",
            current_action.based_on_judgment.value,
            cloud_analysis.judgment.value,
        )
        if cloud_analysis.judgment == Judgment.NORMAL:
            return ControlAction(
                type="maintain",
                based_on_judgment=Judgment.NORMAL,
            )
        elif cloud_analysis.judgment == Judgment.ALARM:
            return self._alarm_action(cloud_analysis)
        else:
            return self._conservative_action(cloud_analysis)

    # -------------------------------------------------------------------

    @staticmethod
    def _emergency_action(analysis: AnalysisResult) -> ControlAction:
        return ControlAction(
            type="emergency_stop",
            params={"reason": analysis.reasoning or "critical_safety"},
            based_on_judgment=Judgment.ALARM,
        )

    @staticmethod
    def _conservative_action(analysis: AnalysisResult) -> ControlAction:
        action_map = {
            "adjust_flow": {"valve_delta": -0.1},
            "reduce_input": {"input_rate_factor": 0.8},
            "increase_cooling": {"cooling_delta": +0.15},
        }
        params = action_map.get(analysis.suggested_action, {"valve_delta": -0.05})
        return ControlAction(
            type=analysis.suggested_action or "adjust_flow",
            params=params,
            based_on_judgment=analysis.judgment,
        )

    @staticmethod
    def _alarm_action(analysis: AnalysisResult) -> ControlAction:
        return ControlAction(
            type="emergency_stop",
            params={
                "reason": analysis.reasoning or "alarm",
                "notify_operator": True,
            },
            based_on_judgment=Judgment.ALARM,
        )
