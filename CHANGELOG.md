# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `scripts/evaluate_cambridge.py score` now defaults to a robust
  IRLS Sim(3) fit between our SfM and the dataset's NVM frame.
  When auditing the v0.2.0 results we noticed a handful of train
  images were mis-registered in our SfM (2 / 231 on ShopFacade,
  13 / 895 on Old Hospital) and biased the alignment. With the
  outliers dropped, the published median errors tighten:
    - ShopFacade: 0.93°/0.21 m → **0.61°/0.096 m** (0.49% → 0.23%)
    - Old Hospital: 1.11°/0.88 m → **1.11°/0.85 m** (1.42% → 1.37%)
  Use `--no-robust-sim3` to reproduce the legacy LS behaviour.
- README at-a-glance and detail sections updated with the
  robust-Sim(3) numbers.

## [0.2.0] - 2026-05-01

Headline change: ROS2 nodes now reject visually-plausible-but-spatially-
implausible poses before publishing them, and there are two more public
datasets verified end-to-end.

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
- pytest `testpaths` extended to also collect
  `ros2/visual_map_localizer_ros/test/`; the rclpy-touching tests
  guard themselves with `pytest.importorskip` so they skip cleanly
  on stock CI.

### Changed
- README no longer cites hardware-specific wall times or GPU model
  names. Per-frame latency is replaced with a short prose section on
  the architectural levers (subprocess vs persistent instance, ndarray
  vs path, resolution scaling). Accuracy numbers are unchanged.

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

[Unreleased]: https://github.com/rsasaki0109/visual-map-localizer/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rsasaki0109/visual-map-localizer/releases/tag/v0.2.0
[0.1.0]: https://github.com/rsasaki0109/visual-map-localizer/releases/tag/v0.1.0
