"""VM side: decompress the camera stream, track the wrist, solve arm IK.

Inputs  (from the notebook): /link/color/compressed, /link/depth/compressedDepth,
                              /camera/camera/color/camera_info
Outputs (to the notebook):   /wrist_pose, /tf (torso_link -> wrist_target),
                              /g1_visualization/joint_states, /link/debug/compressed
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from g1_teleop.urdf_info import rest_tip_position, right_arm_geometry


def _nodes(context):
    urdf_file = os.path.join(get_package_share_directory('g1_teleop'), 'urdf',
                             LaunchConfiguration('model').perform(context))
    shoulder_xyz, reach = right_arm_geometry(urdf_file)
    # Where the robot's wrist is with every joint at 0: the arm is held there until calibrated.
    rest_wrist = rest_tip_position(urdf_file)

    decompress_color = Node(
        package='image_transport', executable='republish', name='decompress_color',
        parameters=[{'in_transport': 'compressed', 'out_transport': 'raw'}],
        remappings=[('in/compressed', '/link/color/compressed'), ('out', '/stream/color/image_raw')])
    decompress_depth = Node(
        package='image_transport', executable='republish', name='decompress_depth',
        parameters=[{'in_transport': 'compressedDepth', 'out_transport': 'raw'}],
        remappings=[('in/compressedDepth', '/link/depth/compressedDepth'), ('out', '/stream/depth/image_raw')])
    detector = Node(
        package='g1_teleop', executable='wrist_detector', name='wrist_detector',
        output='screen', emulate_tty=True,
        parameters=[{'color_topic': '/stream/color/image_raw',
                     'depth_topic': '/stream/depth/image_raw',
                     'camera_info_topic': '/camera/camera/color/camera_info',
                     'show_window': False, 'output_frame': 'torso_link',
                     'robot_reach': reach, 'scale_factor': reach / 0.65,
                     'shoulder_offset': shoulder_xyz,
                     'calibration_required': LaunchConfiguration('calibrate').perform(context).lower()
                                             in ('true', '1', 'yes'),
                     'rest_target': rest_wrist}])
    ik = Node(package='g1_teleop', executable='g1_arm_ik_node', name='g1_arm_ik_node',
              output='screen',
              parameters=[{'urdf_path': urdf_file, 'base_frame': 'torso_link',
                           'elbow_weight': float(LaunchConfiguration('elbow_weight').perform(context))}])
    compress_debug = Node(
        package='image_transport', executable='republish', name='compress_debug',
        parameters=[{'in_transport': 'raw', 'out_transport': 'compressed'}],
        remappings=[('in', '/tracking/debug_image'), ('out/compressed', '/link/debug/compressed')])
    return [decompress_color, decompress_depth, detector, ik, compress_debug]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model', default_value='g1_29dof_rev_1_0.urdf',
                              description='URDF filename in g1_teleop/urdf'),
        DeclareLaunchArgument('calibrate', default_value='true',
                              description='Require a rest-pose calibration before tracking starts'),
        DeclareLaunchArgument('elbow_weight', default_value='0.01',
                              description='How strongly the forearm direction shapes the arm '
                                          'posture. 0 = wrist-only IK.'),
        OpaqueFunction(function=_nodes),
    ])
