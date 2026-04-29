"""Verify the ndarray helpers used by `localize(image_np, camera=...)`."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from visual_map_localizer.localization.pipeline import (  # noqa: E402
    _is_image_array,
    _write_image_array,
)


def test_is_image_array_accepts_2d_and_3d():
    assert _is_image_array(np.zeros((10, 10), dtype=np.uint8))
    assert _is_image_array(np.zeros((10, 10, 3), dtype=np.uint8))


def test_is_image_array_rejects_other_shapes():
    assert not _is_image_array(np.zeros(()))
    assert not _is_image_array(np.zeros((1, 2, 3, 4)))
    assert not _is_image_array("not an array")
    assert not _is_image_array(None)


def test_write_image_array_roundtrip(tmp_path: Path):
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    rgb[..., 0] = 255  # pure red in RGB convention
    out = tmp_path / "out.png"
    _write_image_array(rgb, out)
    assert out.exists()

    # Re-load via cv2 (BGR) and confirm channel order was actually swapped.
    bgr = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert bgr.shape == (20, 30, 3)
    # cv2 reads BGR, so a RED RGB pixel should have B=0, G=0, R=255 → in BGR
    # array that's (0, 0, 255).
    assert tuple(bgr[0, 0]) == (0, 0, 255)


def test_write_image_array_handles_float(tmp_path: Path):
    img = np.full((4, 4, 3), 200.7, dtype=np.float32)
    out = tmp_path / "out.png"
    _write_image_array(img, out)
    assert out.exists()


def test_write_image_array_rejects_weird_channels(tmp_path: Path):
    bad = np.zeros((4, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="unexpected image array shape"):
        _write_image_array(bad, tmp_path / "x.png")
