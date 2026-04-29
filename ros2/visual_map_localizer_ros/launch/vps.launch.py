"""Launch the VPS node with parameters exposed on the command line.

Example::

    ros2 launch visual_map_localizer_ros vps.launch.py \\
        map_dir:=/abs/path/to/map \\
        image_topic:=/my_camera/image_rect \\
        camera_info_topic:=/my_camera/camera_info \\
        publish_tf:=true \\
        frame_id:=map \\
        child_frame_id:=camera_optical_frame
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_arg(name: str, default: str, description: str = "") -> DeclareLaunchArgument:
    return DeclareLaunchArgument(name, default_value=default, description=description)


def generate_launch_description() -> LaunchDescription:
    args = [
        _launch_arg("map_dir", "",
                    "Directory produced by `visual-map-localizer build-map`."),
        _launch_arg("image_topic", "/camera/image_raw",
                    "sensor_msgs/Image input topic."),
        _launch_arg("camera_info_topic", "/camera/camera_info",
                    "sensor_msgs/CameraInfo input topic."),
        _launch_arg("pose_topic", "/vps_pose",
                    "geometry_msgs/PoseWithCovarianceStamped output topic."),
        _launch_arg("frame_id", "map",
                    "Reference frame id of the published pose."),
        _launch_arg("child_frame_id", "camera_optical_frame",
                    "Camera frame id (used for the optional TF broadcast)."),
        _launch_arg("publish_tf", "false",
                    "If true, broadcast frame_id -> child_frame_id TF."),
        _launch_arg("use_camera_info", "true",
                    "If false, ignore CameraInfo and use static intrinsics."),
        _launch_arg("top_k", "10",
                    "Number of retrieval candidates."),
        _launch_arg("ransac_max_error_px", "12.0",
                    "RANSAC reprojection threshold in pixels."),
        _launch_arg("min_inliers", "12",
                    "Minimum number of PnP inliers for success."),
        _launch_arg("cov_base_pos", "0.10",
                    "Base position 1-sigma (metres) at sqrt(inliers)=1."),
        _launch_arg("cov_base_rot_deg", "5.0",
                    "Base rotation 1-sigma (deg) at sqrt(inliers)=1."),
    ]

    node = Node(
        package="visual_map_localizer_ros",
        executable="vps_node",
        name="vps_node",
        output="screen",
        emulate_tty=True,
        parameters=[{
            "map_dir":             LaunchConfiguration("map_dir"),
            "image_topic":         LaunchConfiguration("image_topic"),
            "camera_info_topic":   LaunchConfiguration("camera_info_topic"),
            "pose_topic":          LaunchConfiguration("pose_topic"),
            "frame_id":            LaunchConfiguration("frame_id"),
            "child_frame_id":      LaunchConfiguration("child_frame_id"),
            "publish_tf":          LaunchConfiguration("publish_tf"),
            "use_camera_info":     LaunchConfiguration("use_camera_info"),
            "top_k":               LaunchConfiguration("top_k"),
            "ransac_max_error_px": LaunchConfiguration("ransac_max_error_px"),
            "min_inliers":         LaunchConfiguration("min_inliers"),
            "cov_base_pos":        LaunchConfiguration("cov_base_pos"),
            "cov_base_rot_deg":    LaunchConfiguration("cov_base_rot_deg"),
        }],
    )

    return LaunchDescription([*args, node])
