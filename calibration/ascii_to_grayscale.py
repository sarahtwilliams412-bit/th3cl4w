"""
ASCII-to-Grayscale Converter & Density LUT

Converts ASCII character grids (uint8 codepoints) into grayscale images
suitable for OpenCV corner detection. Also exports the density lookup
table used by the visual hull reconstructor.

The density LUT maps each printable ASCII character (codepoints 32-126)
to a visual density value in [0.0, 1.0] based on the character's
approximate pixel coverage when rendered in a monospace font.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Character density ramp from the existing ASCII converter
# Ordered from sparse (space) to dense (@)
CHARSET_STANDARD = " .:-=+*#%@"

# Extended 70-char ramp for finer density gradation
CHARSET_DETAILED = (
    " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
)

# Approximate visual density for all printable ASCII (32-126)
# Based on pixel coverage in standard monospace fonts.
# Characters not in either ramp get interpolated estimates.
_CHAR_DENSITY_MAP: dict[str, float] = {}


def _build_density_map() -> dict[str, float]:
    """Build density mapping for all printable ASCII characters."""
    density = {}

    # Use the detailed charset as primary reference
    for i, ch in enumerate(CHARSET_DETAILED):
        density[ch] = i / (len(CHARSET_DETAILED) - 1)

    # Fill in remaining printable ASCII with reasonable estimates
    # based on typical glyph pixel coverage
    _extra = {
        " ": 0.0,
        "!": 0.15,
        '"': 0.12,
        "#": 0.75,
        "$": 0.82,
        "%": 0.78,
        "&": 0.80,
        "'": 0.05,
        "(": 0.18,
        ")": 0.18,
        "*": 0.40,
        "+": 0.30,
        ",": 0.08,
        "-": 0.12,
        ".": 0.04,
        "/": 0.20,
        "0": 0.60,
        "1": 0.20,
        "2": 0.50,
        "3": 0.48,
        "4": 0.45,
        "5": 0.50,
        "6": 0.55,
        "7": 0.35,
        "8": 0.70,
        "9": 0.55,
        ":": 0.10,
        ";": 0.12,
        "<": 0.22,
        "=": 0.25,
        ">": 0.22,
        "?": 0.30,
        "@": 0.90,
        "A": 0.55,
        "B": 0.65,
        "C": 0.45,
        "D": 0.60,
        "E": 0.50,
        "F": 0.42,
        "G": 0.55,
        "H": 0.55,
        "I": 0.20,
        "J": 0.30,
        "K": 0.50,
        "L": 0.30,
        "M": 0.75,
        "N": 0.60,
        "O": 0.58,
        "P": 0.45,
        "Q": 0.62,
        "R": 0.52,
        "S": 0.48,
        "T": 0.35,
        "U": 0.50,
        "V": 0.40,
        "W": 0.78,
        "X": 0.48,
        "Y": 0.35,
        "Z": 0.48,
        "[": 0.22,
        "\\": 0.20,
        "]": 0.22,
        "^": 0.10,
        "_": 0.15,
        "`": 0.05,
        "a": 0.45,
        "b": 0.52,
        "c": 0.35,
        "d": 0.52,
        "e": 0.42,
        "f": 0.28,
        "g": 0.48,
        "h": 0.48,
        "i": 0.15,
        "j": 0.18,
        "k": 0.45,
        "l": 0.15,
        "m": 0.62,
        "n": 0.45,
        "o": 0.45,
        "p": 0.50,
        "q": 0.50,
        "r": 0.25,
        "s": 0.38,
        "t": 0.28,
        "u": 0.42,
        "v": 0.32,
        "w": 0.58,
        "x": 0.38,
        "y": 0.35,
        "z": 0.35,
        "{": 0.22,
        "|": 0.15,
        "}": 0.22,
        "~": 0.18,
    }

    # Merge, preferring detailed charset values
    for ch, d in _extra.items():
        if ch not in density:
            density[ch] = d

    return density


_CHAR_DENSITY_MAP = _build_density_map()


def build_density_lut() -> np.ndarray:
    """Build the density lookup table indexed by ASCII codepoint.

    Returns
    -------
    np.ndarray
        float32[128] where lut[codepoint] = density in [0.0, 1.0].
        Non-printable characters (0-31, 127) map to 0.0.
    """
    lut = np.zeros(128, dtype=np.float32)
    for codepoint in range(32, 127):
        ch = chr(codepoint)
        lut[codepoint] = _CHAR_DENSITY_MAP.get(ch, 0.0)
    return lut


# Pre-built LUT for fast access
DENSITY_LUT = build_density_lut()


def ascii_grid_to_grayscale(grid: np.ndarray) -> np.ndarray:
    """Convert a uint8 ASCII codepoint grid to a grayscale image.

    Parameters
    ----------
    grid : np.ndarray
        uint8 array of shape [H, W] containing ASCII codepoints.

    Returns
    -------
    np.ndarray
        float32 array of shape [H, W] in range [0, 255], suitable
        for OpenCV processing.
    """
    # Clip to valid LUT range
    clipped = np.clip(grid, 0, 127)
    # Vectorized lookup
    density = DENSITY_LUT[clipped]  # [H, W] in [0, 1]
    # Scale to [0, 255] for OpenCV
    grayscale = (density * 255.0).astype(np.float32)
    return grayscale


def save_density_lut(path: str | Path) -> None:
    """Save the density LUT to a .npy file.

    Parameters
    ----------
    path : str or Path
        Output file path (should end in .npy).
    """
    np.save(str(path), DENSITY_LUT)
    logger.info("Density LUT saved to %s", path)


def load_density_lut(path: str | Path) -> np.ndarray:
    """Load a density LUT from a .npy file.

    Parameters
    ----------
    path : str or Path
        Path to .npy file.

    Returns
    -------
    np.ndarray
        float32[128] density lookup table.
    """
    lut = np.load(str(path))
    assert lut.shape == (128,), f"Expected shape (128,), got {lut.shape}"
    return lut.astype(np.float32)
