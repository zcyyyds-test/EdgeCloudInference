"""Image loading utilities for multimodal anomaly detection.

Supports two modes:
1. MVTec AD dataset: real industrial images with ground truth labels
2. Synthetic: generates text descriptions when real images are not available
"""

from __future__ import annotations

import base64
import random
from dataclasses import dataclass
from pathlib import Path

from edgerouter.core.schema import Difficulty, Judgment, ScenarioProfile


# ---------------------------------------------------------------------------
# MVTec AD category → scenario mapping
# ---------------------------------------------------------------------------

# Each MVTec category maps to a difficulty range and judgment
MVTEC_CATEGORY_MAP = {
    # Object categories
    "bottle": {"type": "object", "defects": ["broken_large", "broken_small", "contamination"]},
    "cable": {"type": "object", "defects": ["bent_wire", "cable_swap", "combined", "cut_inner_insulation", "cut_outer_insulation", "missing_cable", "missing_wire", "poke_insulation"]},
    "capsule": {"type": "object", "defects": ["crack", "faulty_imprint", "poke", "scratch", "squeeze"]},
    "hazelnut": {"type": "object", "defects": ["crack", "cut", "hole", "print"]},
    "metal_nut": {"type": "object", "defects": ["bent", "color", "flip", "scratch"]},
    "pill": {"type": "object", "defects": ["color", "combined", "contamination", "crack", "faulty_imprint", "scratch", "type"]},
    "screw": {"type": "object", "defects": ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"]},
    "toothbrush": {"type": "object", "defects": ["defective"]},
    "transistor": {"type": "object", "defects": ["bent_lead", "cut_lead", "damaged_case", "misplaced"]},
    "zipper": {"type": "object", "defects": ["broken_teeth", "combined", "fabric_border", "fabric_interior", "rough", "split_teeth", "squeezed_teeth"]},
    # Texture categories
    "carpet": {"type": "texture", "defects": ["color", "cut", "hole", "metal_contamination", "thread"]},
    "grid": {"type": "texture", "defects": ["bent", "broken", "glue", "metal_contamination", "thread"]},
    "leather": {"type": "texture", "defects": ["color", "cut", "fold", "glue", "poke"]},
    "tile": {"type": "texture", "defects": ["crack", "glue_strip", "gray_stroke", "oil", "rough"]},
    "wood": {"type": "texture", "defects": ["color", "combined", "hole", "liquid", "scratch"]},
}


@dataclass
class ImageSample:
    """A single image sample with metadata."""
    path: str                     # absolute path to image file
    category: str                 # MVTec category (e.g. "bottle")
    split: str                    # "train" or "test"
    label: str                    # "good" or defect type (e.g. "crack")
    is_defective: bool            # True if defective
    difficulty: Difficulty         # mapped difficulty
    ground_truth_judgment: Judgment


def _map_to_difficulty(label: str) -> tuple[Difficulty, Judgment]:
    """Map MVTec label to EdgeRouter difficulty and judgment."""
    if label == "good":
        return Difficulty.NORMAL, Judgment.NORMAL
    # All defects are anomalous or critical depending on severity
    return Difficulty.ANOMALOUS, Judgment.ALARM


class MVTecLoader:
    """Load images from MVTec AD dataset directory.

    Expected directory structure:
        data/mvtec/
        ├── bottle/
        │   ├── train/good/
        │   ├── test/good/
        │   └── test/broken_large/
        ├── cable/
        │   └── ...
        └── ...
    """

    def __init__(self, root_dir: str = "data/mvtec"):
        self.root = Path(root_dir)

    @property
    def available(self) -> bool:
        """Check if MVTec dataset is downloaded."""
        return self.root.exists() and any(self.root.iterdir())

    def list_categories(self) -> list[str]:
        """List available MVTec categories."""
        if not self.root.exists():
            return []
        return sorted(d.name for d in self.root.iterdir() if d.is_dir())

    def load_samples(
        self,
        categories: list[str] | None = None,
        split: str = "test",
        max_per_category: int | None = None,
        seed: int = 42,
    ) -> list[ImageSample]:
        """Load image samples from the dataset.

        Args:
            categories: MVTec categories to load. None = all available.
            split: "train" or "test"
            max_per_category: limit samples per category (for quick testing)
            seed: random seed for sampling
        """
        rng = random.Random(seed)
        samples: list[ImageSample] = []

        cats = categories or self.list_categories()
        for cat in cats:
            cat_dir = self.root / cat / split
            if not cat_dir.exists():
                continue

            cat_samples: list[ImageSample] = []
            for label_dir in sorted(cat_dir.iterdir()):
                if not label_dir.is_dir():
                    continue
                label = label_dir.name
                difficulty, judgment = _map_to_difficulty(label)

                for img_path in sorted(label_dir.glob("*.png")):
                    cat_samples.append(ImageSample(
                        path=str(img_path),
                        category=cat,
                        split=split,
                        label=label,
                        is_defective=(label != "good"),
                        difficulty=difficulty,
                        ground_truth_judgment=judgment,
                    ))

            if max_per_category and len(cat_samples) > max_per_category:
                cat_samples = rng.sample(cat_samples, max_per_category)

            samples.extend(cat_samples)

        return samples

    def sample_to_scenario(self, sample: ImageSample) -> ScenarioProfile:
        """Convert an MVTec image sample to a ScenarioProfile."""
        # Map defective/normal to appropriate anomaly levels
        if sample.is_defective:
            anomaly_level = random.uniform(60.0, 90.0)
            secondary = random.uniform(0.3, 0.7)
            texture = random.uniform(0.2, 0.6)
        else:
            anomaly_level = random.uniform(40.0, 60.0)
            secondary = random.uniform(0.05, 0.15)
            texture = random.uniform(0.0, 0.1)

        return ScenarioProfile(
            name=f"MVTec/{sample.category}/{sample.label}",
            difficulty=sample.difficulty,
            true_anomaly_level=anomaly_level,
            true_secondary_metric=secondary,
            true_texture_irregularity=texture,
            ground_truth_judgment=sample.ground_truth_judgment,
            description=f"MVTec AD: {sample.category} - {sample.label}",
        )


def encode_image_base64(image_path: str) -> str:
    """Read an image file and return base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
