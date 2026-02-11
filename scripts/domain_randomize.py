#!/usr/bin/env python3
"""Domain randomization pipeline for sim-to-real transfer.

Takes NVIDIA Kitchen-Sim-Demos MuJoCo scenes and applies domain
randomization to generate augmented training data for robust
policy transfer to the real D1 arm.

Randomizes:
- Lighting (position, intensity, color temperature)
- Material textures and colors
- Camera viewpoints (within ±5° of calibrated)
- Object positions (within ±50mm on table surface)
- Table/counter material

Usage:
    python -m scripts.domain_randomize \
        --scene data/nvidia_kitchen/extracted/OpenCabinet/lerobot/extras/episode_000000/model.xml.gz \
        --variants 100 \
        --output data/augmented/OpenCabinet/
"""

import argparse
import gzip
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class DomainRandomizer:
    """Applies domain randomization to MuJoCo scene XML files."""

    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)

    def randomize_lighting(self, xml: str) -> str:
        """Randomize light positions and intensities."""
        # Randomize directional light
        pos = self._rng.uniform([-2, -2, 2], [2, 2, 5])
        diffuse = self._rng.uniform(0.3, 1.0, size=3)
        specular = self._rng.uniform(0.0, 0.5, size=3)

        # Replace or append light element
        light_xml = (
            f'<light name="dr_light" pos="{pos[0]:.2f} {pos[1]:.2f} {pos[2]:.2f}" '
            f'diffuse="{diffuse[0]:.2f} {diffuse[1]:.2f} {diffuse[2]:.2f}" '
            f'specular="{specular[0]:.2f} {specular[1]:.2f} {specular[2]:.2f}" '
            f'directional="true" castshadow="true"/>'
        )

        # Add ambient light variation
        ambient = self._rng.uniform(0.1, 0.4, size=3)

        # Insert before </worldbody>
        if "</worldbody>" in xml:
            xml = xml.replace("</worldbody>", f"  {light_xml}\n</worldbody>")

        return xml

    def randomize_materials(self, xml: str) -> str:
        """Randomize material colors with controlled variation."""
        def _jitter_rgba(match):
            rgba = match.group(1).split()
            if len(rgba) >= 3:
                jittered = []
                for i, v in enumerate(rgba[:3]):
                    val = float(v) + self._rng.uniform(-0.15, 0.15)
                    jittered.append(f"{np.clip(val, 0, 1):.3f}")
                # Keep alpha
                alpha = rgba[3] if len(rgba) > 3 else "1"
                return f'rgba="{" ".join(jittered)} {alpha}"'
            return match.group(0)

        return re.sub(r'rgba="([^"]+)"', _jitter_rgba, xml)

    def randomize_camera(self, xml: str, max_angle_deg: float = 5.0) -> str:
        """Randomize camera viewpoints within controlled bounds."""
        def _jitter_pos(match):
            pos = [float(x) for x in match.group(1).split()]
            if len(pos) == 3:
                # Small perturbation
                jitter = self._rng.uniform(-0.05, 0.05, size=3)
                pos = [p + j for p, j in zip(pos, jitter)]
                return f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"'
            return match.group(0)

        # Only jitter camera positions, not body positions
        lines = xml.split("\n")
        result = []
        in_camera = False
        for line in lines:
            if "<camera" in line:
                in_camera = True
            if in_camera and 'pos="' in line:
                line = re.sub(r'pos="([^"]+)"', _jitter_pos, line)
            if in_camera and "/>" in line:
                in_camera = False
            result.append(line)
        return "\n".join(result)

    def randomize_object_positions(self, xml: str, max_offset_m: float = 0.05) -> str:
        """Randomize positions of manipulable objects."""
        def _jitter_body_pos(match):
            pos = [float(x) for x in match.group(1).split()]
            if len(pos) == 3:
                # Only jitter x and y (keep height stable)
                pos[0] += self._rng.uniform(-max_offset_m, max_offset_m)
                pos[1] += self._rng.uniform(-max_offset_m, max_offset_m)
                return f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"'
            return match.group(0)

        return re.sub(r'pos="([^"]+)"', _jitter_body_pos, xml)

    def generate_variant(self, base_xml: str, variant_idx: int) -> str:
        """Generate a single randomized variant of the scene."""
        self._rng = np.random.RandomState(variant_idx * 1000 + 42)

        xml = base_xml
        xml = self.randomize_lighting(xml)
        xml = self.randomize_materials(xml)
        xml = self.randomize_camera(xml)
        # Note: object position randomization is optional and can break
        # task semantics, so it's disabled by default
        return xml

    def generate_variants(
        self,
        base_xml: str,
        n_variants: int,
        output_dir: Path,
    ) -> List[Path]:
        """Generate multiple randomized variants and save to disk.

        Returns list of output file paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []

        for i in range(n_variants):
            variant_xml = self.generate_variant(base_xml, i)
            out_path = output_dir / f"variant_{i:04d}.xml"
            with open(out_path, "w") as f:
                f.write(variant_xml)
            paths.append(out_path)

        logger.info("Generated %d variants in %s", n_variants, output_dir)
        return paths


def load_scene_xml(scene_path: str) -> str:
    """Load a scene XML, decompressing if .gz."""
    path = Path(scene_path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return f.read()
    else:
        with open(path) as f:
            return f.read()


def main():
    parser = argparse.ArgumentParser(
        description="Domain randomization for sim-to-real transfer"
    )
    parser.add_argument(
        "--scene", required=True,
        help="Path to MuJoCo scene XML (or .xml.gz)",
    )
    parser.add_argument(
        "--variants", type=int, default=100,
        help="Number of randomized variants to generate",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for variants",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    scene_xml = load_scene_xml(args.scene)
    randomizer = DomainRandomizer(seed=args.seed)
    paths = randomizer.generate_variants(
        scene_xml,
        args.variants,
        Path(args.output),
    )
    logger.info("Done. Generated %d variants.", len(paths))


if __name__ == "__main__":
    main()
