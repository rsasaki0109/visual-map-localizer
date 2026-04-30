# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Apache-2.0 `LICENSE` file.
- GitHub Actions CI: unit-test matrix (Python 3.10 / 3.11 / 3.12) and ruff lint.
- `[deep]` optional extra in `pyproject.toml` for `torch` / `torchvision`,
  keeping the base install lightweight.
- README badges (CI status, license, Python, COLMAP, hloc, ROS2),
  Mermaid pipeline diagram, table of contents, Quick Start block,
  and an at-a-glance benchmark header.
- `docs/assets/demo.gif` — animated demo showing 10 south-building
  queries with their recovered cameras visualized in 3D.
- `scripts/render_demo.py` — generator for the demo GIF
  (matplotlib + Pillow + imageio + pycolmap).

### Changed
- `torch` / `torchvision` are no longer base dependencies. Install them
  explicitly with `pip install .[deep]` when running the real
  retrieval / matching pipeline.

## [0.1.0] - 2026-04-30

### Added
- Initial implementation of `visual-map-localizer`:
  - SfM-based map building via `pycolmap` + `hloc`
    (NetVLAD retrieval, DISK local features, LightGlue matcher).
  - `VisualMapLocalizer` Python API for single-image 6DoF localization,
    accepting both file paths and `np.ndarray` BGR / RGB queries.
  - PnP+RANSAC backend with `pycolmap` primary path and OpenCV fallback;
    full COLMAP camera-model support
    (`SIMPLE_PINHOLE` / `PINHOLE` / `SIMPLE_RADIAL` / `OPENCV` /
    `FULL_OPENCV` / `OPENCV_FISHEYE`).
  - CLI: `build-map`, `localize`, `inspect`.
  - JSON output schema for downstream consumption.
- ROS2 wrapper package `visual_map_localizer_ros`:
  - `vps_node` subscribing to `sensor_msgs/Image` (+ optional
    `CameraInfo`) and publishing `geometry_msgs/PoseStamped` on
    `/vps_pose`, plus optional TF.
  - Frame-drop strategy with single-threaded executor.
  - cv_bridge-free image conversion (works with NumPy 2.x on Jazzy).
- Documentation:
  - `README.md` with quick-start, dataset walk-through, ROS2 usage.
  - `docs/` with extended design notes.
- Evaluation + profiling scripts under `scripts/`.
- 17 unit tests (`tests/`) + 5 ROS2 helper tests
  (`ros2/visual_map_localizer_ros/test/`).

### Verified
- South-Building public dataset: 10/10 query images successfully
  localized; median rotation error 0.066°, median translation error
  0.034 % of scene extent.
- ROS2 end-to-end: identical accuracy via `/camera/image_raw` →
  `/vps_pose` round-trip.
- Persistent-process latency: 4.4 s warm-up, 1.41 s steady-state
  per frame on the test machine.

[Unreleased]: https://github.com/rsasaki0109/visual-map-localizer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rsasaki0109/visual-map-localizer/releases/tag/v0.1.0
