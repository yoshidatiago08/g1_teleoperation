"""Notebook side: show the tracking overlay and the simulated G1 in RViz.

robot_state_publisher runs here (not on the VM) so the URDF and meshes never cross the
network; only the small joint-state and TF messages do.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    pkg = get_package_share_directory('g1_teleop')
    with open(os.path.join(pkg, 'urdf', LaunchConfiguration('model').perform(context))) as f:
        robot_description = f.read()
    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[{'robot_description': robot_description}],
               remappings=[('/joint_states', '/g1_visualization/joint_states')])
    decompress_debug = Node(
        package='image_transport', executable='republish', name='decompress_debug',
        parameters=[{'in_transport': 'compressed', 'out_transport': 'raw'}],
        remappings=[('in/compressed', '/link/debug/compressed'), ('out', '/tracking/debug_image_view')])
    rviz = Node(package='rviz2', executable='rviz2', output='screen',
                arguments=['-d', os.path.join(pkg, 'config', 'g1_wrist.rviz')])
    return [rsp, decompress_debug, rviz]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model', default_value='g1_29dof_inspire_dfq.urdf',
                              description='URDF filename in g1_teleop/urdf'),
        OpaqueFunction(function=_nodes),
    ])
