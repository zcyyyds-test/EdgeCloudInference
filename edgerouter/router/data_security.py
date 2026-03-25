"""Industrial data security: classify and sanitize data for cloud transmission."""

from __future__ import annotations

from edgerouter.core.schema import ProcessContext, SecurityLevel, VisionOutput


class DataSecurityChecker:
    """Check whether process context contains data that must stay on edge.

    Industrial data security is field-level: recipe params, customer info,
    and reaction conditions are confidential and must never leave the factory.
    """

    def contains_sensitive(self, context: ProcessContext) -> bool:
        """Return True if context contains any confidential data."""
        return (
            context.has_recipe_params
            or context.has_customer_info
            or context.has_reaction_params
        )

    def classify(self, context: ProcessContext) -> SecurityLevel:
        """Return the highest security level present in the context."""
        if context.has_recipe_params or context.has_customer_info or context.has_reaction_params:
            return SecurityLevel.CONFIDENTIAL
        if context.equipment_id or context.batch_id:
            return SecurityLevel.INTERNAL
        return SecurityLevel.PUBLIC

    def sanitize_for_cloud(
        self,
        vision_output: VisionOutput,
        context: ProcessContext,
    ) -> dict:
        """Return only cloud-safe fields (no raw images, no process params)."""
        return {
            "anomaly_level": round(vision_output.anomaly_level, 2),
            "secondary_metric": round(vision_output.secondary_metric, 3),
            "texture_irregularity": round(vision_output.texture_irregularity, 3),
            "surface_uniformity": round(vision_output.surface_uniformity, 3),
            "anomaly_score": round(vision_output.anomaly_score, 3),
            "anomaly_confidence": round(vision_output.anomaly_confidence, 3),
            "trend_summary": context.get_trend_summary(),
            # Explicitly excluded: equipment_id, batch_id, recipe, customer, reaction params
        }
