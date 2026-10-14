"""VM side: decompress the camera stream, track the wrist, solve arm IK.

Inputs  (from the notebook): /link/color/compressed, /link/depth/compressedDepth,
                              /camera/camera/color/camera_info
Outputs (to the notebook):   /<side>/wrist_pose, /<side>/hand_state, /waist_state, /<side>/elbow_pose, /tf (torso_upright ->
                              <side>_wrist_target, <side>_elbow_target),
                              /g1_visualization/joint_states, /link/debug/compressed
(<side> is right and left; the `arms` argument picks which ones are tracked.)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from g1_teleop.urdf_info import arm_geometry, rest_tip_position


def _nodes(context):
    urdf_file = os.path.join(get_package_share_directory('g1_teleop'), 'urdf',
                             LaunchConfiguration('model').perform(context))
    sides = [s for s in LaunchConfiguration('arms').perform(context).replace(',', ' ').split() if s]
    detector_params = {'camera_tilt_deg': float(LaunchConfiguration('camera_tilt').perform(context)),
                       'waist_enabled': LaunchConfiguration('waist').perform(context).lower()
                                        in ('true', '1', 'yes'),
                       'hands_enabled': LaunchConfiguration('hands').perform(context).lower()
                                         in ('true', '1', 'yes'),
                       'color_topic': '/stream/color/image_raw',
                       'depth_topic': '/stream/depth/image_raw',
                       'camera_info_topic': '/camera/camera/color/camera_info',
                       'show_window': False, 'output_frame': 'torso_upright', 'arms': sides,
                       'calibration_required': LaunchConfiguration('calibrate').perform(context).lower()
                                               in ('true', '1', 'yes')}
    reaches = []
    for side in sides:
        shoulder_xyz, reach = arm_geometry(urdf_file, side)
        reaches.append(reach)
        detector_params.update({f'{side}_shoulder_offset': shoulder_xyz,
                                f'{side}_robot_reach': reach,
                                # Where the robot's wrist is with every joint at 0: the arm is
                                # held there until calibrated.
                                f'{side}_rest_target': rest_tip_position(urdf_file, side)})
    detector_params['scale_factor'] = reaches[0] / 0.65  # fallback until calibrated

    decompress_color = Node(
        package='image_transport', executable='republish', name='decompress_color',
        parameters=[{'in_transport': 'compressed', 'out_transport': 'raw'}],
        remappings=[('in/compressed', '/link/color/compressed'), ('out', '/stream/color/image_raw')])
    decompress_depth = Node(
        package='image_transport', executable='republish', name='decompress_depth',
        parameters=[{'in_transport': 'compressedDepth', 'out_transport': 'raw'}],
        remappings=[('in/compressedDepth', '/link/depth/compressedDepth'), ('out', '/stream/depth/image_raw')])
    detector = Node(package='g1_teleop', executable='wrist_detector', name='wrist_detector',
                    output='screen', emulate_tty=True, parameters=[detector_params])
    ik = Node(package='g1_teleop', executable='g1_arm_ik_node', name='g1_arm_ik_node',
              output='screen',
              parameters=[{'urdf_path': urdf_file, 'base_frame': 'torso_link', 'sides': sides,
                           'elbow_weight': float(LaunchConfiguration('elbow_weight').perform(context)),
                           'orientation_weight': float(
                               LaunchConfiguration('orientation_weight').perform(context))}])
    compress_debug = Node(
        package='image_transport', executable='republish', name='compress_debug',
        parameters=[{'in_transport': 'raw', 'out_transport': 'compressed'}],
        remappings=[('in', '/tracking/debug_image'), ('out/compressed', '/link/debug/compressed')])
    return [decompress_color, decompress_depth, detector, ik, compress_debug]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model', default_value='g1_29dof_inspire_dfq.urdf',
                              description='URDF filename in g1_teleop/urdf'),
        DeclareLaunchArgument('arms', default_value='right,left',
                              description="Arms to track: 'right', 'left' or 'right,left'"),
        DeclareLaunchArgument('calibrate', default_value='true',
                              description='Require a rest-pose calibration before tracking starts'),
        DeclareLaunchArgument('camera_tilt', default_value='0.0',
                              description='Degrees the camera looks down (positive) from level'),
        DeclareLaunchArgument('waist', default_value='true',
                              description="Move the robot's waist with the operator's torso"),
        DeclareLaunchArgument('hands', default_value='true',
                              description='Track the palm orientation and finger curls'),
        DeclareLaunchArgument('orientation_weight', default_value='0.01',
                              description='How strongly the wrist follows your palm orientation. '
                                          '0 = position only.'),
        DeclareLaunchArgument('elbow_weight', default_value='0.01',
                              description='How strongly the forearm direction shapes the arm '
                                          'posture. 0 = wrist-only IK.'),
        OpaqueFunction(function=_nodes),
    ])
