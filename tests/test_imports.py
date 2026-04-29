"""Smoke-test that every module imports without optional heavy deps."""
from __future__ import annotations
import importlib

import pytest


SUBMODULES = [
    "visual_map_localizer",
    "visual_map_localizer.config",
    "visual_map_localizer.io",
    "visual_map_localizer.io.output",
    "visual_map_localizer.io.camera",
    "visual_map_localizer.localization.pnp",
    "visual_map_localizer.cli.main",
]


@pytest.mark.parametrize("name", SUBMODULES)
def test_import(name: str):
    importlib.import_module(name)
