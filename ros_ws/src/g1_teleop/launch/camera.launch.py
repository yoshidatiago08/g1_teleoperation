"""Notebook side: the camera, compressed for the trip to the VM.

The RealSense driver (or the fake camera) publishes raw images on the local machine only.
Two `republish` nodes compress them onto /link/*, which is what crosses the network.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'])),
        launch_arguments={'align_depth.enable': 'true',
                          'rgb_camera.color_profile': '640x480x30',
                          'depth_module.depth_profile': '640x480x30'}.items(),
        condition=UnlessCondition(LaunchConfiguration('fake')))
    fake = Node(package='g1_teleop', executable='fake_camera', output='screen',
                condition=IfCondition(LaunchConfiguration('fake')))
    compress_color = Node(
        package='image_transport', executable='republish', name='compress_color',
        parameters=[{'in_transport': 'raw', 'out_transport': 'compressed'}],
        remappings=[('in', '/camera/camera/color/image_raw'), ('out/compressed', '/link/color/compressed')])
    compress_depth = Node(
        package='image_transport', executable='republish', name='compress_depth',
        parameters=[{'in_transport': 'raw', 'out_transport': 'compressedDepth'}],
        remappings=[('in', '/camera/camera/aligned_depth_to_color/image_raw'),
                    ('out/compressedDepth', '/link/depth/compressedDepth')])
    return LaunchDescription([
        DeclareLaunchArgument('fake', default_value='false',
                              description='Use a synthetic camera instead of a RealSense'),
        realsense, fake, compress_color, compress_depth,
    ])
