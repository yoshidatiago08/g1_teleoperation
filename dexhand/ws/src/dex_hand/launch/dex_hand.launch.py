"""The whole thing in one command: the tracking node, robot_state_publisher (draws the hand) and RViz.

    ros2 launch dex_hand dex_hand.launch.py hand:=right source:=webcam motion_scale:=0.3
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = {  # name: (default, description)
    'hand': ('right', 'right or left'),
    'source': ('webcam', 'webcam | realsense | synthetic (a made-up hand: no camera needed)'),
    'device': ('0', 'webcam index, device path or video file'),
    'motion_scale': ('0.3', 'robot hand motion per unit of your hand motion (0.3 = it moves 30%)'),
    'rotation_scale': ('1.0', 'same for rotation (1.0 = it turns like your hand)'),
    'finger_scale': ('1.15', "dex-retargeting scaling_factor: your fingers' size vs the robot's"),
    'hfov_deg': ('70.0', 'webcam horizontal field of view (degrees), used to place your hand in 3D'),
    'show_window': ('true', 'camera window with the clutch and scale keys'),
    'rviz': ('true', 'start RViz'),
}


def _nodes(context):
    pkg = get_package_share_directory('dex_hand')
    arg = {name: LaunchConfiguration(name).perform(context) for name in ARGS}
    with open(os.path.join(pkg, 'urdf', f'inspire_hand_{arg["hand"]}.urdf')) as f:
        robot_description = f.read()
    params = {'hand': arg['hand'], 'source': arg['source'], 'device': arg['device'],
              'motion_scale': float(arg['motion_scale']), 'rotation_scale': float(arg['rotation_scale']),
              'finger_scale': float(arg['finger_scale']), 'hfov_deg': float(arg['hfov_deg']),
              'show_window': arg['show_window'].lower() in ('true', '1', 'yes')}
    return [
        Node(package='dex_hand', executable='dex_hand', name='dex_hand', output='screen',
             emulate_tty=True, parameters=[params]),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description}]),
        Node(package='rviz2', executable='rviz2', output='screen', condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', os.path.join(pkg, 'config', 'dex_hand.rviz')]),
    ]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument(n, default_value=d, description=h) for n, (d, h) in ARGS.items()]
                             + [OpaqueFunction(function=_nodes)])
