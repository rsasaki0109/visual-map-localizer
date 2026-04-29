"""ROS2 node that wraps `VisualMapLocalizer`.

Subscriptions
-------------
* ``image_topic`` (default ``/camera/image_raw``) — ``sensor_msgs/Image``
* ``camera_info_topic`` (default ``/camera/camera_info``) — ``sensor_msgs/CameraInfo``
  (optional; falls back to ``static_camera_*`` parameters when omitted)

Publications
------------
* ``pose_topic`` (default ``/vps_pose``) — ``geometry_msgs/PoseWithCovarianceStamped``
  (camera pose in ``frame_id``)
* TF (optional, controlled by ``publish_tf``) — ``frame_id -> child_frame_id``

Pose convention
---------------
* COLMAP / pycolmap and ``visual_map_localizer`` use the
  *camera-from-world* (R, t) convention.
* ROS PoseStamped expects the *world-from-camera* convention (i.e. the camera
  pose AS SEEN IN the world frame). This node therefore inverts the pose
  before publishing.

Frame drop policy
-----------------
The localizer takes ~1 s per frame on a modern GPU; ROS cameras typically
publish at 30 Hz. To avoid an unbounded backlog we keep an in-flight flag and
silently drop incoming frames while the previous localization is still being
processed. The latest dropped frame is not retried — it is simply lost. This
matches the "absolute pose at low Hz" use case for which VPS is intended.
"""
from __future__ import annotations
from pathlib import Path
import threading
import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import (
    PoseWithCovariance,
    PoseWithCovarianceStamped,
    TransformStamped,
)
from std_msgs.msg import Header

from visual_map_localizer import VisualMapLocalizer
from visual_map_localizer.config import LocalizeConfig


# --------------------------------------------------------------------- helpers
def _quat_from_rotmat(R: np.ndarray) -> np.ndarray:
    """3x3 -> (x, y, z, w) — the order ROS uses."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def _image_msg_to_rgb_array(msg: Image) -> np.ndarray:
    """Decode a `sensor_msgs/Image` to an HxWx3 uint8 RGB array.

    We deliberately avoid `cv_bridge` because the cv_bridge shipped with
    ROS2 Jazzy is built against numpy 1.x and segfaults under numpy >= 2.
    Only the encodings we actually publish from typical cameras are
    supported here.
    """
    enc = (msg.encoding or "").lower()
    h, w = int(msg.height), int(msg.width)
    buf = bytes(msg.data)
    if enc == "rgb8":
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        return arr.copy()
    if enc == "bgr8":
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        return np.ascontiguousarray(arr[..., ::-1])
    if enc == "rgba8":
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return arr[..., :3].copy()
    if enc == "bgra8":
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return np.ascontiguousarray(arr[..., 2::-1])
    if enc in ("mono8", "8uc1"):
        gray = np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
        return np.repeat(gray[..., None], 3, axis=2)
    raise ValueError(f"unsupported image encoding: {msg.encoding!r}")


def _pycolmap_camera_from_camera_info(msg: CameraInfo):
    """Best-effort `pycolmap.Camera` from a ROS CameraInfo message.

    Maps the ROS distortion models we know about onto COLMAP names. Falls back
    to PINHOLE (distortion ignored) for unknown models.
    """
    import pycolmap

    fx = msg.k[0]
    fy = msg.k[4]
    cx = msg.k[2]
    cy = msg.k[5]
    w = int(msg.width)
    h = int(msg.height)
    model = (msg.distortion_model or "").lower()
    d = list(msg.d)

    if model in ("plumb_bob", "rational_polynomial") and len(d) >= 4:
        # ROS plumb_bob uses k1, k2, t1, t2, k3 (k3 may be missing)
        k1, k2, p1, p2 = d[:4]
        k3 = d[4] if len(d) >= 5 else 0.0
        if any(d):
            if k3:
                return pycolmap.Camera(
                    model="FULL_OPENCV", width=w, height=h,
                    params=[fx, fy, cx, cy, k1, k2, p1, p2, k3, 0, 0, 0],
                )
            return pycolmap.Camera(
                model="OPENCV", width=w, height=h,
                params=[fx, fy, cx, cy, k1, k2, p1, p2],
            )
    if model == "equidistant" and len(d) >= 4:
        k1, k2, k3, k4 = d[:4]
        return pycolmap.Camera(
            model="OPENCV_FISHEYE", width=w, height=h,
            params=[fx, fy, cx, cy, k1, k2, k3, k4],
        )
    return pycolmap.Camera(
        model="PINHOLE", width=w, height=h, params=[fx, fy, cx, cy],
    )


# --------------------------------------------------------------------- node
class VpsNode(Node):
    def __init__(self) -> None:
        super().__init__("vps_node")

        # -------- parameters
        self.declare_parameter("map_dir", "")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("pose_topic", "/vps_pose")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("child_frame_id", "camera_optical_frame")
        self.declare_parameter("top_k", 10)
        self.declare_parameter("ransac_max_error_px", 12.0)
        self.declare_parameter("min_inliers", 12)
        self.declare_parameter("publish_tf", False)
        self.declare_parameter("use_camera_info", True)

        # Static camera fallback parameters (used when CameraInfo is unavailable).
        self.declare_parameter("static_camera_model", "")  # empty = disabled
        self.declare_parameter("static_camera_params", [0.0])
        self.declare_parameter("static_image_size", [0, 0])

        # Heuristic covariance: σ_pos = base_pos / sqrt(inliers); σ_rot likewise.
        self.declare_parameter("cov_base_pos", 0.10)  # metres for sqrt(inliers)=1
        self.declare_parameter("cov_base_rot_deg", 5.0)

        map_dir = self.get_parameter("map_dir").value
        if not map_dir:
            raise RuntimeError("'map_dir' parameter is required")
        if not Path(map_dir).exists():
            raise RuntimeError(f"map_dir does not exist: {map_dir}")

        cfg = LocalizeConfig(
            top_k=int(self.get_parameter("top_k").value),
            ransac_max_error_px=float(self.get_parameter("ransac_max_error_px").value),
            ransac_min_inliers=int(self.get_parameter("min_inliers").value),
        )
        self.get_logger().info(f"loading map from {map_dir} ...")
        t = time.perf_counter()
        self.localizer = VisualMapLocalizer(map_dir, config=cfg)
        self.get_logger().info(
            f"map ready ({self.localizer.map.num_images} imgs / "
            f"{self.localizer.map.num_points} points3D) "
            f"in {time.perf_counter() - t:.1f}s"
        )

        # -------- IO
        self._busy_lock = threading.Lock()
        self._busy = False
        self._frame_counter = 0
        self._success_counter = 0
        self._dropped_counter = 0

        self._pycolmap_camera = self._build_static_camera_or_none()
        self._camera_info_received = self._pycolmap_camera is not None

        # -------- publishers
        pose_topic = self.get_parameter("pose_topic").value
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, pose_topic, 10,
        )

        if bool(self.get_parameter("publish_tf").value):
            from tf2_ros import TransformBroadcaster
            self.tf_broadcaster = TransformBroadcaster(self)
        else:
            self.tf_broadcaster = None

        # -------- subscriptions
        image_topic = self.get_parameter("image_topic").value
        self.image_sub = self.create_subscription(
            Image, image_topic, self._on_image, qos_profile_sensor_data,
        )
        if bool(self.get_parameter("use_camera_info").value):
            camera_info_topic = self.get_parameter("camera_info_topic").value
            self.camera_info_sub = self.create_subscription(
                CameraInfo, camera_info_topic, self._on_camera_info,
                qos_profile_sensor_data,
            )

        self.get_logger().info(
            f"VPS node up. listening on {image_topic}, publishing to {pose_topic}"
        )

    # ------------------------------------------------------------ camera info
    def _build_static_camera_or_none(self):
        model = str(self.get_parameter("static_camera_model").value or "").strip()
        if not model:
            return None
        params = list(self.get_parameter("static_camera_params").value or [])
        size = list(self.get_parameter("static_image_size").value or [0, 0])
        if len(size) != 2 or any(s <= 0 for s in size):
            self.get_logger().warning(
                "static_camera_model is set but static_image_size is invalid; "
                "ignoring the static camera."
            )
            return None
        import pycolmap
        return pycolmap.Camera(
            model=model, width=int(size[0]), height=int(size[1]), params=params,
        )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        # Only update once unless dimensions / intrinsics actually change.
        try:
            cam = _pycolmap_camera_from_camera_info(msg)
        except Exception as exc:  # pragma: no cover
            self.get_logger().warning(f"failed to parse CameraInfo: {exc}")
            return
        self._pycolmap_camera = cam
        if not self._camera_info_received:
            self._camera_info_received = True
            self.get_logger().info(
                f"got camera intrinsics: {cam.model_name} "
                f"{cam.width}x{cam.height} params={list(cam.params)}"
            )

    # ------------------------------------------------------------ image cb
    def _on_image(self, msg: Image) -> None:
        if self._pycolmap_camera is None:
            self.get_logger().throttle(2.0).warning(
                "no camera intrinsics yet — waiting for CameraInfo "
                "or set static_camera_* parameters"
            )
            return

        with self._busy_lock:
            if self._busy:
                self._dropped_counter += 1
                return
            self._busy = True

        try:
            try:
                rgb = _image_msg_to_rgb_array(msg)
            except Exception as exc:
                self.get_logger().error(f"image decode failed: {exc}")
                return

            self._frame_counter += 1
            t0 = time.perf_counter()
            res = self.localizer.localize(
                rgb,
                camera=self._pycolmap_camera,
                name=f"frame_{self._frame_counter:06d}.png",
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0

            if not res.success:
                self.get_logger().info(
                    f"frame {self._frame_counter}: FAILED ({res.error}) "
                    f"in {dt_ms:.0f} ms"
                )
                return

            self._success_counter += 1
            self._publish_pose(msg.header, res, dt_ms)
        finally:
            with self._busy_lock:
                self._busy = False

    # ------------------------------------------------------------ publish
    def _publish_pose(self, image_header: Header, res, dt_ms: float) -> None:
        # res.pose is camera-from-world (R_cw, t_cw). ROS expects world-from-cam.
        R_cw = np.asarray(res.pose["R"])
        t_cw = np.asarray(res.pose["t"])
        # camera-in-world = (-R_cw^T t_cw, R_cw^T)
        R_wc = R_cw.T
        C_wc = -R_wc @ t_cw
        q = _quat_from_rotmat(R_wc)  # (x, y, z, w)

        cov = self._estimate_covariance(res)

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = image_header.stamp
        msg.header.frame_id = self.get_parameter("frame_id").value
        pwc = PoseWithCovariance()
        pwc.pose.position.x = float(C_wc[0])
        pwc.pose.position.y = float(C_wc[1])
        pwc.pose.position.z = float(C_wc[2])
        pwc.pose.orientation.x = float(q[0])
        pwc.pose.orientation.y = float(q[1])
        pwc.pose.orientation.z = float(q[2])
        pwc.pose.orientation.w = float(q[3])
        pwc.covariance = cov.flatten().tolist()
        msg.pose = pwc
        self.pose_pub.publish(msg)

        if self.tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp = msg.header.stamp
            tf.header.frame_id = msg.header.frame_id
            tf.child_frame_id = self.get_parameter("child_frame_id").value
            tf.transform.translation.x = float(C_wc[0])
            tf.transform.translation.y = float(C_wc[1])
            tf.transform.translation.z = float(C_wc[2])
            tf.transform.rotation.x = float(q[0])
            tf.transform.rotation.y = float(q[1])
            tf.transform.rotation.z = float(q[2])
            tf.transform.rotation.w = float(q[3])
            self.tf_broadcaster.sendTransform(tf)

        self.get_logger().info(
            f"frame {self._frame_counter}: OK inliers={res.inliers} "
            f"reproj={res.reproj_error:.2f}px in {dt_ms:.0f}ms "
            f"(success {self._success_counter}, dropped {self._dropped_counter})"
        )

    def _estimate_covariance(self, res) -> np.ndarray:
        """Diagonal heuristic — replace with a Hessian-based estimate later.

        Scales the per-axis std with 1 / sqrt(inliers) so high-confidence
        localizations get small covariances. The base scales come from
        parameters and should be tuned per scene.
        """
        base_pos = float(self.get_parameter("cov_base_pos").value)
        base_rot_deg = float(self.get_parameter("cov_base_rot_deg").value)
        scale = 1.0 / max(np.sqrt(max(res.inliers, 1)), 1.0)
        sigma_p = base_pos * scale
        sigma_r = np.deg2rad(base_rot_deg) * scale
        cov = np.zeros((6, 6), dtype=np.float64)
        cov[0, 0] = cov[1, 1] = cov[2, 2] = sigma_p ** 2
        cov[3, 3] = cov[4, 4] = cov[5, 5] = sigma_r ** 2
        return cov


# --------------------------------------------------------------------- main
def main(args=None) -> None:
    rclpy.init(args=args)
    node = VpsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
