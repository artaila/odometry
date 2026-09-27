import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory("tram_navigation")
    params_file = os.path.join(pkg_share, "config", "params.yaml")

    return LaunchDescription([
        Node(
            package="tram_navigation",
            executable="telemetry_synchronizer_node",
            name="tram_backup_odometry",
            parameters=[params_file],
            output="screen"
        )
    ])
