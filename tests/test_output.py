"""LocalizationResult JSON formatting."""
from __future__ import annotations
import json

import numpy as np

from visual_map_localizer.io.output import LocalizationResult


def test_failure_to_json_roundtrip():
    res = LocalizationResult.failure("no inliers", retrieval=["a.jpg"])
    parsed = json.loads(res.to_json())
    assert parsed["success"] is False
    assert parsed["error"] == "no inliers"
    assert parsed["retrieval"] == ["a.jpg"]


def test_success_to_json_contains_pose():
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    res = LocalizationResult.from_pose(
        R, t,
        inliers=42,
        reproj_error=0.7,
        num_matches=128,
        retrieval=["a.jpg", "b.jpg"],
        query="q.jpg",
    )
    parsed = json.loads(res.to_json())
    assert parsed["success"] is True
    assert parsed["pose"]["R"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert parsed["pose"]["t"] == [1.0, 2.0, 3.0]
    assert parsed["inliers"] == 42
    assert parsed["num_matches"] == 128
    assert parsed["query"] == "q.jpg"
