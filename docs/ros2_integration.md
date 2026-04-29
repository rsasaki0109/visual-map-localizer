# ROS2 統合 (実装済み — 設計ノート)

> **実装は `ros2/visual_map_localizer_ros/` 配下にあります。**
> このドキュメントはノードの設計判断・座標系・拡張ポイントの解説です。
> 実際のビルド方法やパラメータ一覧は
> [`ros2/visual_map_localizer_ros/README.md`](../ros2/visual_map_localizer_ros/README.md) を参照。


## ノードイメージ

```
        /camera/image_raw                        /vps_pose
          (sensor_msgs/Image)  ──▶ vps_node ──▶ (geometry_msgs/PoseWithCovarianceStamped)
                                       │
                                       └──▶ /vps_diagnostics (vision_msgs/Detection2DArray など)
```

## ノードスケッチ

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge

from visual_map_localizer import VisualMapLocalizer
from visual_map_localizer.config import LocalizeConfig


class VpsNode(Node):
    def __init__(self):
        super().__init__("vps_node")
        self.declare_parameter("map_dir", "")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("top_k", 10)
        self.declare_parameter("ransac_max_error", 12.0)

        map_dir = self.get_parameter("map_dir").value
        cfg = LocalizeConfig(
            top_k=int(self.get_parameter("top_k").value),
            ransac_max_error_px=float(self.get_parameter("ransac_max_error").value),
        )
        self.localizer = VisualMapLocalizer(map_dir, config=cfg)
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, "/camera/image_raw", self.cb, 1)
        self.pub = self.create_publisher(PoseWithCovarianceStamped, "/vps_pose", 1)

    def cb(self, msg: Image):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        # save to tmpfile; or extend the localizer to accept np.ndarray.
        ...
```

## TF / 座標系

* `vps_pose` は world → camera (COLMAP の慣例) なので、
  ROS の `map → base_link` を取りたいときは事前に外部キャリブレーション
  (`base_link` ↔ `camera`) を掛ける。
* `frame_id` は地図構築時のワールド座標系名 (例 `map`)。
* `child_frame_id` は `base_link` か `camera_optical_frame`。

## 位置不確かさ

- `inliers` と `reproj_error` から経験的に共分散を推定:
  - σ_pos ≈ k1 * reproj_error / sqrt(inliers)
  - σ_rot ≈ k2 / sqrt(inliers)
- もしくは Hessian ベースの推定 (pycolmap で使える場合) を使う。

## VIO / GNSS / IMU 融合

- `VisualMapLocalizer.localize()` の出力を `robot_localization` などの
  EKF / UKF パイプラインに観測として供給。
- VPS は低頻度 (1〜5Hz)・絶対姿勢、VIO/IMU は高頻度・相対姿勢として併用するのが基本。
- **失敗時の取り扱い**: `success=False` のフレームは EKF に送らないか、
  共分散を非常に大きくして実質無視する。

## リアルタイム化のロードマップ

1. **特徴抽出を起動毎ではなく常駐化**: `pipeline.py` を画像配列で受け付ける形にし、
   tmp ファイル経由を廃止する。
2. **GPU 推論の常駐化**: SuperPoint / LightGlue モデルをノード初期化時にロード。
3. **検索高速化**: NetVLAD の DB descriptors を Faiss / FAISS-GPU に持たせ、
   `pairs_from_retrieval` を ANN 化。
4. **DB 分割 (sharding)**: 大規模地図はタイル単位で分割し、粗い位置 (GNSS/odom) で
   タイル選択 → そのタイル内で fine localization。
