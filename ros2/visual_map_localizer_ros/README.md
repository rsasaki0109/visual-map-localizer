# visual_map_localizer_ros

`visual-map-localizer` を ROS2 ノードとして公開する薄いラッパーです。

## 機能

| Topic / TF | 型 | 方向 | 説明 |
|---|---|---|---|
| `/camera/image_raw`     | `sensor_msgs/Image`    | sub | 入力画像 |
| `/camera/camera_info`   | `sensor_msgs/CameraInfo` | sub | 内部パラメータ (省略可・static で代替可) |
| `/vps_pose`             | `geometry_msgs/PoseWithCovarianceStamped` | pub | 推定 6DoF (camera-in-`frame_id`) |
| `frame_id → child_frame_id` | TF | pub | `publish_tf:=true` のとき |

座標系は **ROS 慣例 (world-from-camera)**。COLMAP の camera-from-world を
ノード内で反転して publish しています。

## ビルド (ROS2 Jazzy 想定)

```bash
# 1) Python ライブラリ側 (= visual-map-localizer 本体) を先に入れる
cd <repo-root>
pip install --user --break-system-packages -e .
pip install --user --break-system-packages git+https://github.com/cvg/Hierarchical-Localization.git@master

# 2) ROS2 ワークスペースに symlink して colcon build
mkdir -p ~/ros2_ws/src
ln -s <repo-root>/ros2/visual_map_localizer_ros ~/ros2_ws/src/
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select visual_map_localizer_ros --symlink-install
source install/setup.bash
```

## 実行

### 最小構成 (CameraInfo を使う一般的なケース)

```bash
ros2 launch visual_map_localizer_ros vps.launch.py \
    map_dir:=/abs/path/to/map \
    image_topic:=/camera/image_raw \
    camera_info_topic:=/camera/camera_info \
    publish_tf:=true
```

### CameraInfo が出ないカメラ向け (intrinsics を直書き)

```bash
ros2 run visual_map_localizer_ros vps_node \
    --ros-args \
    -p map_dir:=/abs/path/to/map \
    -p use_camera_info:=false \
    -p static_camera_model:=PINHOLE \
    -p static_camera_params:='[600.0, 600.0, 320.0, 240.0]' \
    -p static_image_size:='[640, 480]'
```

## パラメータ一覧

| パラメータ | 既定 | 説明 |
|---|---|---|
| `map_dir` | (必須) | `build-map` の出力ディレクトリ |
| `image_topic` | `/camera/image_raw` | 入力画像トピック |
| `camera_info_topic` | `/camera/camera_info` | CameraInfo トピック |
| `pose_topic` | `/vps_pose` | 出力 PoseWithCovarianceStamped |
| `frame_id` | `map` | publish 時の reference frame |
| `child_frame_id` | `camera_optical_frame` | TF の子フレーム |
| `publish_tf` | `false` | TF を publish するか |
| `use_camera_info` | `true` | CameraInfo を購読するか |
| `static_camera_model` | `""` | 例: `PINHOLE`, `SIMPLE_RADIAL` (空なら CameraInfo 必須) |
| `static_camera_params` | `[]` | モデルに対応する焦点距離・主点・歪み係数 |
| `static_image_size` | `[0, 0]` | `[width, height]` (static 利用時のみ) |
| `top_k` | `10` | retrieval Top-K |
| `ransac_max_error_px` | `12.0` | PnP RANSAC 閾値 (px) |
| `min_inliers` | `12` | success 判定の閾値 |
| `cov_base_pos` | `0.10` | 共分散ヒューリスティクス: 位置 base σ (m) |
| `cov_base_rot_deg` | `5.0` | 共分散ヒューリスティクス: 回転 base σ (deg) |
| `outlier_max_linear_velocity_mps` | `10.0` | これを超える線速度を含意する pose は破棄 (≤0 で無効) |
| `outlier_max_angular_velocity_dps` | `60.0` | これを超える角速度を含意する pose は破棄 (≤0 で無効) |
| `outlier_state_timeout_sec` | `30.0` | この秒数以上 pose 受理がないと gate state をリセット (再アンカー) |

## 設計ノート

### Frame drop 戦略

ノードはローカライズ中フラグ `_busy` を持ち、in-flight 中に届いた画像は
**捨てます**。VPS は 1 Hz 程度の遅い絶対姿勢源として使うのが想定で、
30 Hz カメラの全フレームを処理するのは現実的ではないため。

実装は単スレッド executor 前提。マルチスレッド化したい場合は
`MultiThreadedExecutor` + 排他制御に差し替え可能。

### 共分散の扱い

現状はヒューリスティクス (`σ ∝ 1 / sqrt(inliers)`) のため絶対精度では
ありません。`robot_localization` 等に流す場合は計測実験の上で
`cov_base_pos` / `cov_base_rot_deg` を調整してください。Hessian ベースの
推定を入れたい場合は `_estimate_covariance` を差し替えるだけです。

### Outlier rejection (pose gate)

成功した localize 結果でも、稀に **建物の別の階・別の似たファサードに
match が引っ掛かって** 同じ inlier 数で全く違う pose を返すことがあります。
inlier や reproj error だけでは見抜けないので、直前 accept した pose との
**線速度 / 角速度** を計算し、上限を超える場合は publish しないようにしています。

* 既定値 (`10 m/s` / `60 deg/s`) は地上ロボット・手持ちカメラには十分緩く、
  典型的な VPS 失敗 (建物別フロアにジャンプ) は確実に弾けるレベル。
* gate ロジックは `pose_gate.PoseGate` に切り出してあり、`rclpy` 非依存で
  `test_pose_gate.py` から単体テスト可能。
* 長時間無受信 (例: ノードが一旦止まった) で過去の anchor が古くなった場合は
  `outlier_state_timeout_sec` 秒で state をリセットして再アンカーします。
* まったく無効化したい場合は `outlier_max_linear_velocity_mps:=0.0`
  (どちらかの閾値を `<=0` にすると gate 全体が disable)。

### Pose 規約

* `visual_map_localizer.LocalizationResult.pose["R", "t"]` は **camera-from-world** (COLMAP)
* `geometry_msgs/Pose.position / orientation` は **world-from-camera** (ROS)
* ノード内部で `R_wc = R_cw.T`, `C_wc = -R_wc @ t_cw` に変換した上で publish

### VIO / GNSS 融合

* この VPS は遅いが絶対姿勢として、上位 EKF (例 `robot_localization`) に
  `geometry_msgs/PoseWithCovarianceStamped` として食わせる想定
* IMU / VIO は高頻度・相対姿勢として併走させる
* 失敗フレーム (`success=false`) は publish しないので、フィルタ側で安全
