"""Parse ASCII art text into a 2D binary occupancy grid.

Each character cell is classified as *filled* (part of the object silhouette)
or *empty* (background).  The parser handles common ASCII-art conventions:
  - Space, dot, and empty cells → empty
  - Any other printable character → filled
  - Lines are padded to uniform width so the grid is rectangular
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np

# Characters treated as "empty" (background)
_EMPTY_CHARS: Set[str] = {" ", ".", ""}

# Characters that act as annotation labels rather than geometry
_LABEL_MARKERS: Set[str] = {"(", ")", "<", ">", "-", "=", "~"}


@dataclass
class AsciiImage:
    """A parsed 2D binary grid derived from ASCII art.

    Attributes
    ----------
    grid : np.ndarray
        Boolean 2D array — ``True`` where the character is filled.
        Shape is (height, width) with row-0 at the *top* of the text.
    raw_lines : list[str]
        Original text lines (for debug / round-trip).
    """

    grid: np.ndarray  # shape (H, W), dtype bool
    raw_lines: List[str]

    @property
    def height(self) -> int:
        return self.grid.shape[0]

    @property
    def width(self) -> int:
        return self.grid.shape[1]

    def filled_coords(self) -> List[Tuple[int, int]]:
        """Return list of (row, col) tuples for every filled cell."""
        rows, cols = np.nonzero(self.grid)
        return list(zip(rows.tolist(), cols.tolist()))

    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Return (min_row, min_col, max_row, max_col) of filled region."""
        rows, cols = np.nonzero(self.grid)
        if len(rows) == 0:
            return (0, 0, 0, 0)
        return (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))


def _classify_char(ch: str) -> bool:
    """Return True if *ch* represents a filled (solid) cell."""
    if ch in _EMPTY_CHARS:
        return False
    return True


def parse_ascii(text: str, trim: bool = True) -> AsciiImage:
    """Parse a multiline ASCII-art string into an `AsciiImage`.

    Parameters
    ----------
    text : str
        The raw ASCII art, e.g. read from a ``.txt`` file.
    trim : bool
        If True, remove fully-empty leading/trailing rows and columns so the
        grid tightly bounds the artwork.

    Returns
    -------
    AsciiImage
    """
    lines = text.rstrip("\n").split("\n")

    # Pad all lines to the same width
    max_width = max((len(l) for l in lines), default=0)
    lines = [l.ljust(max_width) for l in lines]

    height = len(lines)
    width = max_width

    grid = np.zeros((height, width), dtype=bool)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            grid[r, c] = _classify_char(ch)

    if trim:
        grid, lines = _trim(grid, lines)

    return AsciiImage(grid=grid, raw_lines=lines)


def parse_file(path: str | Path, trim: bool = True) -> AsciiImage:
    """Read an ASCII art file and parse it."""
    text = Path(path).read_text()
    return parse_ascii(text, trim=trim)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _trim(
    grid: np.ndarray, lines: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Remove empty border rows/columns."""
    rows_any = np.any(grid, axis=1)
    cols_any = np.any(grid, axis=0)

    if not rows_any.any():
        # Entirely empty — return 1×1 empty grid
        return np.zeros((1, 1), dtype=bool), [""]

    r_min, r_max = int(np.argmax(rows_any)), int(len(rows_any) - 1 - np.argmax(rows_any[::-1]))
    c_min, c_max = int(np.argmax(cols_any)), int(len(cols_any) - 1 - np.argmax(cols_any[::-1]))

    grid = grid[r_min : r_max + 1, c_min : c_max + 1]
    lines = [l[c_min : c_max + 1] for l in lines[r_min : r_max + 1]]
    return grid, lines
