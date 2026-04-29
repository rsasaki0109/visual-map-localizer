"""Output formatting for localization results."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import numpy as np


@dataclass
class LocalizationResult:
    """Result of a single-image localization run.

    Designed to be cheaply JSON-serialisable so it doubles as the on-the-wire
    schema printed by the CLI.
    """

    success: bool
    pose: Optional[Dict[str, Any]] = None  # {"R": [[..]], "t": [..], "qvec": [..]}
    inliers: int = 0
    reproj_error: float = float("nan")
    num_matches: int = 0
    retrieval: List[str] = field(default_factory=list)
    error: Optional[str] = None
    timing: Dict[str, float] = field(default_factory=dict)
    query: Optional[str] = None

    # ------------------------------------------------------------------ ctor
    @classmethod
    def from_pose(
        cls,
        R: np.ndarray,
        t: np.ndarray,
        *,
        inliers: int,
        reproj_error: float,
        num_matches: int,
        retrieval: List[str],
        query: Optional[str] = None,
        timing: Optional[Dict[str, float]] = None,
        qvec: Optional[np.ndarray] = None,
    ) -> "LocalizationResult":
        return cls(
            success=True,
            pose={
                "R": np.asarray(R, dtype=np.float64).tolist(),
                "t": np.asarray(t, dtype=np.float64).reshape(-1).tolist(),
                **({"qvec": np.asarray(qvec, dtype=np.float64).tolist()} if qvec is not None else {}),
            },
            inliers=int(inliers),
            reproj_error=float(reproj_error),
            num_matches=int(num_matches),
            retrieval=list(retrieval),
            query=query,
            timing=timing or {},
        )

    @classmethod
    def failure(
        cls,
        error: str,
        *,
        retrieval: Optional[List[str]] = None,
        num_matches: int = 0,
        query: Optional[str] = None,
        timing: Optional[Dict[str, float]] = None,
    ) -> "LocalizationResult":
        return cls(
            success=False,
            error=error,
            retrieval=list(retrieval or []),
            num_matches=int(num_matches),
            query=query,
            timing=timing or {},
        )

    # ------------------------------------------------------------------ json
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)


def dump_json(obj: Any, path: Path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(obj, LocalizationResult):
            f.write(obj.to_json(indent=indent))
        else:
            json.dump(obj, f, indent=indent, default=_json_default)


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
