# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- ROS2 outlier-rejection gate (`pose_gate.PoseGate` + 8 unit tests):
  any localized pose whose implied linear (>10 m/s) or angular
  (>60 deg/s) velocity is unrealistic vs. the last accepted pose is
  logged and dropped instead of published. Three new node parameters:
  `outlier_max_linear_velocity_mps`, `outlier_max_angular_velocity_dps`,
  `outlier_state_timeout_sec`. The gate state resets after a long gap
  to avoid permanently blocking re-localization.
- Cambridge Landmarks ShopFacade end-to-end evaluation
  (`scripts/evaluate_cambridge.py`): 103/103 success, median 0.93° /
  0.21 m (0.49 % of scene).
- Cambridge Landmarks Old Hospital end-to-end evaluation (same script,
  `--scene` swap): 182/182 success, median 1.11° / 0.88 m (1.42 % of
  scene). Larger / longer-day-range scene confirms the pipeline isn't
  tuned to a single capture.
- README at-a-glance header now shows three datasets side by side
  (south-building + Cambridge ShopFacade + Cambridge Old Hospital).

## [0.1.0] - 2026-05-01

First public release.

### Added

#### Core library

- `VisualMapLocalizer` Python API for single-image 6DoF localization,
  accepting both file paths and `np.ndarray` BGR / RGB queries.
- SfM-based map building via `pycolmap` + `hloc`
  (NetVLAD retrieval, DISK / SuperPoint local features, LightGlue matcher).
- PnP + RANSAC backend with `pycolmap` primary path and OpenCV fallback;
  full COLMAP camera-model support
  (`SIMPLE_PINHOLE` / `PINHOLE` / `SIMPLE_RADIAL` / `OPENCV` /
  `FULL_OPENCV` / `OPENCV_FISHEYE`).
- CLI: `build-map`, `localize`, `inspect`.
- JSON output schema for downstream consumption.
- PEP 562 lazy imports — `import visual_map_localizer` works without
  `torch` / `hloc`, so the unit-test surface and JSON I/O can be used
  in lightweight environments.

#### ROS2 wrapper (`visual_map_localizer_ros`)

- `vps_node` subscribing to `sensor_msgs/Image` (+ optional
  `CameraInfo`) and publishing `geometry_msgs/PoseWithCovarianceStamped`
  on `/vps_pose`, plus optional TF broadcast.
- Frame-drop strategy with a single-threaded executor (use as a
  ~1 Hz absolute pose source).
- cv_bridge-free image conversion (works with NumPy 2.x on Jazzy).

#### Packaging / OSS hygiene

- Apache-2.0 `LICENSE`.
- `[deep]` optional extra in `pyproject.toml` for `torch` /
  `torchvision`, keeping the base install lightweight.
- GitHub Actions CI: pytest matrix on Python 3.10 / 3.11 / 3.12
  plus a ruff lint job.
- README with badges, Mermaid pipeline diagram, table of contents,
  Quick Start block, at-a-glance benchmark header, and embedded
  animated demo (`docs/assets/demo.gif`).

#### Examples / scripts

- `examples/build_map_example.py`, `examples/localize_example.py`.
- `scripts/evaluate_south_building.py` — Sim(3) alignment + pose
  error evaluation against a reference SfM.
- `scripts/profile_localize.py` — persistent-process latency
  benchmark.
- `scripts/render_demo.py` — generator for the README demo GIF.

#### Tests

- 17 unit tests under `tests/` (PnP, ndarray cache key, JSON output,
  smoke imports).
- 5 ROS2 helper tests under
  `ros2/visual_map_localizer_ros/test/` (quaternion roundtrip,
  CameraInfo → pycolmap.Camera conversions).

### Verified

- **South-Building** public dataset (118 db / 10 query):
  10/10 success rate; median rotation error 0.066°, median
  translation error 0.034 % of scene extent.
- **ROS2 end-to-end**: identical accuracy via `/camera/image_raw` →
  `/vps_pose` round-trip.

[Unreleased]: https://github.com/rsasaki0109/visual-map-localizer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rsasaki0109/visual-map-localizer/releases/tag/v0.1.0
