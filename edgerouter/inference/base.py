"""Abstract base class for analyzers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from edgerouter.core.schema import AnalysisResult, VisionOutput


class AnalyzerBackend(ABC):
    """Unified interface for edge and cloud analyzers."""

    @abstractmethod
    async def analyze(
        self,
        vision_output: VisionOutput,
        recent_history: list[VisionOutput] | None = None,
        edge_draft: AnalysisResult | None = None,
    ) -> AnalysisResult:
        """Analyze the vision output and return a judgment.

        Parameters
        ----------
        vision_output : VisionOutput
            Current frame detection result.
        recent_history : list[VisionOutput] | None
            Recent frames for temporal context (optional).
        edge_draft : AnalysisResult | None
            Edge analyzer's preliminary judgment (cloud only, for cascade).

        Returns
        -------
        AnalysisResult
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable and ready."""
