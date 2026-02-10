"""
ASCII Video Converter — wraps src.vision.ascii_converter with HTTP frame fetching.
"""

import logging
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError

from src.vision.ascii_converter import (
    AsciiConverter,
    CHARSET_STANDARD,
    CHARSET_DETAILED,
    CHARSET_BLOCKS,
    CHARSET_MINIMAL,
)

logger = logging.getLogger("th3cl4w.ascii.converter")

CAMERA_SERVER_URL = "http://localhost:8081"

CHARSETS = {
    "standard": CHARSET_STANDARD,
    "detailed": CHARSET_DETAILED,
    "blocks": CHARSET_BLOCKS,
    "minimal": CHARSET_MINIMAL,
}


def fetch_jpeg(cam_id: int, timeout: float = 2.0) -> Optional[bytes]:
    """Fetch a JPEG snapshot from the camera server."""
    url = f"{CAMERA_SERVER_URL}/snap/{cam_id}"
    try:
        with urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except (URLError, OSError, TimeoutError) as e:
        logger.debug("Failed to fetch cam %d: %s", cam_id, e)
        return None


def fetch_and_convert(
    cam_id: int,
    width: int = 120,
    height: int = 60,
    charset: str = CHARSET_STANDARD,
    invert: bool = True,
    color: bool = False,
) -> Optional[dict]:
    """Fetch a JPEG frame and convert to ASCII.

    Returns dict with keys: ascii, lines, width, height, cam_id, color_data (if color=True)
    or None if fetch failed.
    """
    jpeg = fetch_jpeg(cam_id)
    if jpeg is None:
        return None

    converter = AsciiConverter(
        width=width, height=height, charset=charset, invert=invert, color=color
    )

    ascii_text = converter.decode_jpeg_to_ascii(jpeg)
    result = {
        "ascii": ascii_text,
        "lines": ascii_text.split("\n"),
        "width": width,
        "height": height,
        "cam_id": cam_id,
    }

    if color:
        result["color_data"] = converter.decode_jpeg_to_color_data(jpeg)

    return result
