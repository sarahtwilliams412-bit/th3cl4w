"""
th3cl4w Vision — ASCII Video Converter

Converts video frames (BGR numpy arrays from OpenCV) into ASCII art.
Supports configurable character sets, output dimensions, and color modes.
Designed to accept frames from CameraThread.get_raw_frame() or any BGR image.
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

# Character ramps ordered from darkest to brightest
CHARSET_STANDARD = " .:-=+*#%@"
CHARSET_DETAILED = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
CHARSET_BLOCKS = " ░▒▓█"
CHARSET_MINIMAL = " .oO@"


class AsciiConverter:
    """Converts BGR video frames to ASCII art representations.

    Parameters
    ----------
    width : int
        Number of characters per output row (default 120).
    height : int
        Number of rows in the output (default 40).
    charset : str
        Character ramp from darkest to brightest.
    invert : bool
        If True, bright pixels map to dense characters (good for dark backgrounds).
    color : bool
        If True, include per-character RGB color data in structured output.
    """

    def __init__(
        self,
        width: int = 120,
        height: int = 40,
        charset: str = CHARSET_STANDARD,
        invert: bool = True,
        color: bool = False,
    ):
        if cv2 is None:
            raise RuntimeError("opencv-python (cv2) is required for AsciiConverter")
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive integers")
        if len(charset) < 2:
            raise ValueError("charset must contain at least 2 characters")
        self.width = width
        self.height = height
        self.charset = charset
        self.invert = invert
        self.color = color

    def frame_to_ascii(self, frame) -> str:
        """Convert a BGR frame to a plain ASCII string.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3) as returned by cv2.imread or CameraThread.get_raw_frame().

        Returns
        -------
        str
            Multi-line ASCII art string.
        """
        gray = self._preprocess(frame)
        return self._map_to_chars(gray)

    def frame_to_color_data(self, frame) -> dict:
        """Convert a BGR frame to structured data with per-character color.

        Returns a dict with:
          - "lines": list of strings (one per row)
          - "colors": list of lists of [r, g, b] per character (only if self.color)
          - "width": output width
          - "height": output height

        Accepts numpy array or cv2.UMat input.
        """
        small = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Convert back to numpy for pixel-level access
        gray = gray.get() if isinstance(gray, cv2.UMat) else gray
        small = small.get() if isinstance(small, cv2.UMat) else small

        lines = []
        colors = []
        n = len(self.charset) - 1

        for y in range(self.height):
            row_chars = []
            row_colors = []
            for x in range(self.width):
                brightness = int(gray[y, x])
                if self.invert:
                    idx = int(brightness / 255 * n)
                else:
                    idx = int((255 - brightness) / 255 * n)
                idx = min(idx, n)
                row_chars.append(self.charset[idx])
                if self.color:
                    b, g, r = small[y, x]
                    row_colors.append([int(r), int(g), int(b)])
            lines.append("".join(row_chars))
            if self.color:
                colors.append(row_colors)

        result: dict = {
            "lines": lines,
            "width": self.width,
            "height": self.height,
        }
        if self.color:
            result["colors"] = colors
        return result

    def _preprocess(self, frame):
        """Resize and convert to grayscale. Accepts numpy array or cv2.UMat."""
        small = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return gray

    def _map_to_chars(self, gray) -> str:
        """Map grayscale pixel values to ASCII characters."""
        gray = gray.get() if isinstance(gray, cv2.UMat) else gray
        n = len(self.charset) - 1
        if self.invert:
            indices = (gray.astype(np.float32) / 255.0 * n).astype(np.int32)
        else:
            indices = ((255 - gray).astype(np.float32) / 255.0 * n).astype(np.int32)
        indices = np.clip(indices, 0, n)

        lines = []
        for row in indices:
            lines.append("".join(self.charset[i] for i in row))
        return "\n".join(lines)

    def decode_jpeg_to_ascii(self, jpeg_bytes: bytes) -> str:
        """Convenience: decode JPEG bytes directly to ASCII string.

        Useful for consuming output from CameraThread.get_frame() which returns
        JPEG-encoded bytes.
        """
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode JPEG bytes")
        return self.frame_to_ascii(frame)

    def decode_jpeg_to_color_data(self, jpeg_bytes: bytes) -> dict:
        """Convenience: decode JPEG bytes to structured color ASCII data."""
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode JPEG bytes")
        return self.frame_to_color_data(frame)
