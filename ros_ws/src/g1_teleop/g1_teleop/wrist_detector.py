#!/usr/bin/env python3
"""
ROS 2 node that tracks the operator's arms with MediaPipe Pose and a depth camera.

For each tracked arm (right and/or left) it finds the shoulder, elbow and wrist, turns them into
3D points with the aligned depth image, expresses them in a body frame anchored at that arm's
shoulder, scales them to the robot's arm, and publishes the targets for the IK node.

Subscribed topics:
    color_topic / depth_topic (sensor_msgs/Image): colour and aligned depth (16UC1 mm or 32FC1 m)
    camera_info_topic (sensor_msgs/CameraInfo): intrinsics of the colour camera

Published topics (per side, 'right' and 'left'):
    /<side>/wrist_pose, /<side>/elbow_pose (geometry_msgs/PoseStamped), in output_frame
    /tf: <side>_wrist_target and <side>_elbow_target
    /tracking/debug_image (sensor_msgs/Image): the overlay
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from std_srvs.srv import Trigger
import tf2_ros
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
import message_filters

# MediaPipe Pose landmark indices (shoulder, elbow, wrist) of each side
ARM_LANDMARKS = {'right': (12, 14, 16), 'left': (11, 13, 15)}
# BGR colours of the shoulder and elbow markers
SHOULDER_COLOR, ELBOW_COLOR = (255, 255, 0), (0, 165, 255)


class ArmState:
    """Everything the detector remembers about one arm."""

    def __init__(self, side, shoulder_offset, robot_reach, rest_target, calibrated):
        self.side = side
        self.tag = side[0].upper()  # 'R' / 'L', used in the overlay
        self.shoulder_idx, self.elbow_idx, self.wrist_idx = ARM_LANDMARKS[side]
        self.shoulder_offset = np.array(shoulder_offset, dtype=float)  # robot shoulder in output frame
        self.robot_reach = float(robot_reach)
        self.rest_target = np.array(rest_target, dtype=float)  # robot wrist at rest (output frame)
        self.calibrated = calibrated
        self.cal_samples = []        # (time, arm length, wrist in body, elbow in body)
        self.cal_message = ''
        # per-frame
        self.shoulder_3d = None      # this frame's shoulder (camera frame), None if not seen
        self.elbow_fresh = False     # True only if the elbow was measured this frame
        self.elbow_seen = False      # elbow landmark visible this frame (overlay)
        # filtered state
        self.filtered_origin = None  # shoulder, camera frame
        self.last_valid_origin = None
        self.filtered_wrist_in_body = None
        self.filtered_elbow_in_body = None
        self.prev_wrist_in_body = None
        # online arm-length estimation (only used when calibration is switched off)
        self.arm_length_samples = []
        self.online_scale = None     # latched scale; None until converged
        self.wrist_pose_pub = None
        self.elbow_pose_pub = None


class WristDetector(Node):
    """ROS 2 node that tracks one or both arms of the operator."""

    def __init__(self):
        super().__init__('wrist_detector')

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Create pose detector with optimized settings for real-time detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.bridge = CvBridge()

        # Camera intrinsics (filled from camera_info)
        self.camera_info = None
        self.fx = self.fy = self.cx = self.cy = None

        # Topics
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('show_window', False)
        self.show_window = self.get_parameter('show_window').value
        self.declare_parameter('show_all_landmarks', False)
        self.declare_parameter('wrist_circle_radius', 10)
        self.declare_parameter('wrist_circle_color', [0, 255, 0])  # Green in BGR
        color_topic = self.get_parameter('color_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.show_all_landmarks = self.get_parameter('show_all_landmarks').value
        self.wrist_radius = self.get_parameter('wrist_circle_radius').value
        self.wrist_color = tuple(self.get_parameter('wrist_circle_color').value)

        # Which arms to track, and the robot geometry of each (the launch file fills these from
        # the URDF). Scale = robot reach / operator arm length, measured in the calibration.
        self.declare_parameter('arms', ['right', 'left'])
        self.declare_parameter('scale_factor', 0.41 / 0.65)  # fallback until calibrated/estimated
        self.scale_factor = self.get_parameter('scale_factor').value
        for side, y in (('right', -0.100), ('left', 0.100)):
            self.declare_parameter(f'{side}_shoulder_offset', [0.004, y, 0.248])
            self.declare_parameter(f'{side}_robot_reach', 0.41)
            self.declare_parameter(f'{side}_rest_target', [0.0, 0.0, 0.0])

        # Online arm-length estimation (only when calibration is off): the sum of the upper-arm
        # and forearm lengths is pose-invariant, so no calibration pose is needed.
        self.declare_parameter('online_scale_estimation', True)
        self.online_scale_estimation = self.get_parameter('online_scale_estimation').value

        # Output frame of the targets (the robot's body frame)
        self.declare_parameter('output_frame', 'body')
        self.output_frame = self.get_parameter('output_frame').value

        # --- Calibration: the operator holds a relaxed rest pose (arms hanging at the sides) so
        # the arm lengths can be measured cleanly. See the README.
        self.declare_parameter('calibration_required', True)
        self.declare_parameter('calibration_seconds', 3.0)   # how long to hold still
        self.declare_parameter('calibration_still_cm', 6.0)  # allowed wrist wobble while holding
        self.calibration_required = self.get_parameter('calibration_required').value
        self.cal_seconds = float(self.get_parameter('calibration_seconds').value)
        self.cal_still = float(self.get_parameter('calibration_still_cm').value) / 100.0

        self.arms = []
        for side in self.get_parameter('arms').value:
            if side not in ARM_LANDMARKS:
                raise RuntimeError(f"arms must be 'right' and/or 'left', got {side!r}")
            arm = ArmState(side,
                           self.get_parameter(f'{side}_shoulder_offset').value,
                           self.get_parameter(f'{side}_robot_reach').value,
                           self.get_parameter(f'{side}_rest_target').value,
                           calibrated=not self.calibration_required)
            # Elbow position in the same frame and scale as the wrist pose. The IK uses the
            # direction elbow -> wrist to choose the arm posture, not the elbow's exact position.
            arm.elbow_pose_pub = self.create_publisher(PoseStamped, f'/{side}/elbow_pose', 10)
            arm.wrist_pose_pub = self.create_publisher(PoseStamped, f'/{side}/wrist_pose', 10)
            self.arms.append(arm)

        # Tracking overlay (skeleton, markers, body axes) as an image topic, so it can be viewed
        # from another machine instead of an OpenCV window.
        self.debug_image_pub = self.create_publisher(Image, '/tracking/debug_image', 1)
        self.create_service(Trigger, 'calibrate', self.calibrate_callback)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Synchronized colour + depth
        self.color_sub = message_filters.Subscriber(self, Image, color_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.synced_callback)

        # Camera info is only needed once
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.camera_info_callback, 10)

        # Statistics
        self.frame_count = 0
        self.detection_count = 0

        # EMA filter parameters
        self.declare_parameter('filter_alpha_axes', 0.15)  # Lower = smoother body frame
        self.declare_parameter('filter_alpha_wrist', 0.2)  # Higher = more responsive wrist
        self.alpha_axes = self.get_parameter('filter_alpha_axes').value
        self.alpha_wrist = self.get_parameter('filter_alpha_wrist').value

        # Jump filters (max allowed movement per frame)
        self.declare_parameter('jump_threshold', 0.40)         # body landmarks, m
        self.declare_parameter('wrist_jump_threshold', 0.40)   # wrist / elbow, m
        self.declare_parameter('axis_jump_threshold_deg', 30.0)
        self.jump_threshold = self.get_parameter('jump_threshold').value
        self.wrist_jump_threshold = self.get_parameter('wrist_jump_threshold').value
        self.axis_jump_threshold = np.radians(self.get_parameter('axis_jump_threshold_deg').value)

        # Shared body frame (EMA-filtered axes)
        self.filtered_axis_x = self.filtered_axis_y = self.filtered_axis_z = None
        self.last_valid_R = None  # body -> camera rotation, from the last frame with no rejection
        self.body_level = 'NONE'  # which landmarks gave the body axes (shown on the overlay)
        self.prev_landmarks_3d = {}  # previous landmark positions for the jump filter
        self.prev_axes = {}          # previous axes for the angular jump filter
        self.body_landmark_rejected = False  # any body landmark rejected this frame

        # Arm-length estimation thresholds (used by the calibration and the online estimate)
        self.arm_length_window = 150       # ~5 s at 30 Hz
        self.arm_length_min_samples = 90   # samples required before latching
        self.arm_length_max_spread = 0.03  # IQR convergence threshold (m)

        # Hand orientation (roll); nothing publishes it yet, so it stays identity
        self.hand_quat = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.hand_roll_sub = self.create_subscription(
            Quaternion, '/hand_roll_quat', self.hand_roll_callback, 10)

        self.get_logger().info('=== Wrist Detector Node Started ===')
        self.get_logger().info(f'Tracking arms: {[a.side for a in self.arms]}; '
                               f'output frame {self.output_frame}')
        self.get_logger().info(f'Subscribing to {color_topic}, {depth_topic}, {camera_info_topic}')
        self.get_logger().info(f'Filter alpha (axes): {self.alpha_axes}, (wrist): {self.alpha_wrist}')
        self.get_logger().info(f'Jump thresholds: body {self.jump_threshold*100:.0f} cm, '
                               f'wrist {self.wrist_jump_threshold*100:.0f} cm, '
                               f'axes {np.degrees(self.axis_jump_threshold):.0f} deg per frame')
        for arm in self.arms:
            self.get_logger().info(f'{arm.side}: robot reach {arm.robot_reach:.3f} m, shoulder '
                                   f'{arm.shoulder_offset}, rest wrist {arm.rest_target}')

    def hand_roll_callback(self, msg):
        """Update hand orientation (roll) from hand_orientation_estimator."""
        self.hand_quat = msg

    def camera_info_callback(self, msg):
        """Store camera info and extract intrinsics."""
        if self.camera_info is None:
            self.camera_info = msg
            # Extract camera intrinsics from K matrix
            # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(f'Received camera info: {msg.width}x{msg.height}')
            self.get_logger().info(f'Intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}')

    def get_depth_at_pixel(self, depth_image, u, v, window_size=5):
        """Get depth at pixel using median of a window to reduce noise."""
        h, w = depth_image.shape[:2]
        half = window_size // 2
        
        # Clamp window to image bounds
        u_min = max(0, u - half)
        u_max = min(w, u + half + 1)
        v_min = max(0, v - half)
        v_max = min(h, v + half + 1)
        
        # Extract window and compute median of valid depths
        window = depth_image[v_min:v_max, u_min:u_max]
        valid_depths = window[(window > 0) & (np.isfinite(window))]
        
        if len(valid_depths) > 0:
            return np.median(valid_depths)
        return None

    def deproject_pixel_to_3d(self, u, v, depth):
        """Convert pixel coordinates + depth to 3D point in camera frame."""
        if self.fx is None or depth is None or depth <= 0:
            return None
        
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        
        return np.array([X, Y, Z])

    def project_3d_to_pixel(self, point_3d):
        """Project 3D point back to pixel coordinates."""
        if self.fx is None or point_3d[2] <= 0:
            return None
        
        u = int(self.fx * point_3d[0] / point_3d[2] + self.cx)
        v = int(self.fy * point_3d[1] / point_3d[2] + self.cy)
        
        return (u, v)

    def apply_ema(self, new_value, filtered_value, alpha):
        """Apply Exponential Moving Average filter.
        
        Args:
            new_value: New measurement (numpy array)
            filtered_value: Previous filtered value (numpy array or None)
            alpha: Filter coefficient (0-1). Higher = more responsive, Lower = smoother
        
        Returns:
            Filtered value
        """
        if filtered_value is None:
            return new_value.copy()
        return alpha * new_value + (1 - alpha) * filtered_value

    def apply_jump_filter(self, landmark_name, new_pos):
        """Apply pure rejection jump filter for body landmarks.
        
        Natural movement at detection frequency always produces intermediate
        points within the threshold. If a reading exceeds the threshold,
        it's bad data (occlusion, depth noise) — reject entirely and keep
        the previous valid position. No convergence toward bad readings.
        
        Args:
            landmark_name: Identifier for the landmark (e.g., 'l_shoulder')
            new_pos: New 3D position (numpy array)
        
        Returns:
            Previous valid position if jump detected, otherwise new_pos
        """
        if landmark_name not in self.prev_landmarks_3d:
            self.prev_landmarks_3d[landmark_name] = new_pos.copy()
            return new_pos
        
        prev_pos = self.prev_landmarks_3d[landmark_name]
        delta = new_pos - prev_pos
        distance = np.linalg.norm(delta)
        
        if distance > self.jump_threshold:
            # Bad data — reject entirely, keep previous valid position
            self.get_logger().warn(
                f'JUMP REJECTED on {landmark_name}: {distance*100:.1f}cm — keeping previous'
            )
            self.body_landmark_rejected = True
            return prev_pos
        else:
            # Normal movement — accept and update baseline
            self.prev_landmarks_3d[landmark_name] = new_pos.copy()
            return new_pos

    # ---------------------------------------------------------------- per-arm scale
    def update_arm_length(self, arm, sh_3d, el_3d, wr_3d):
        """Accumulate arm-length samples and latch the scale on convergence (calibration off).

        ||elbow - shoulder|| + ||wrist - elbow|| is pose-invariant (rigid segments). The median
        of a rolling window rejects depth outliers; once enough samples agree, the scale is
        latched so the mapping does not keep drifting during operation.
        """
        length = np.linalg.norm(el_3d - sh_3d) + np.linalg.norm(wr_3d - el_3d)
        if not (0.3 < length < 1.2):
            return  # implausible arm length: depth artifact

        arm.arm_length_samples.append(length)
        if len(arm.arm_length_samples) > self.arm_length_window:
            arm.arm_length_samples.pop(0)

        if len(arm.arm_length_samples) < self.arm_length_min_samples:
            return

        q1, median, q3 = np.percentile(arm.arm_length_samples, [25, 50, 75])
        if (q3 - q1) < self.arm_length_max_spread:
            arm.online_scale = arm.robot_reach / median
            self.get_logger().info(
                f'{arm.side} arm length converged: {median*100:.1f} cm '
                f'(IQR {(q3-q1)*100:.1f} cm) -> scale latched at {arm.online_scale:.3f}')
        elif self.frame_count % 150 == 0:
            self.get_logger().info(
                f'Estimating {arm.side} arm length: {len(arm.arm_length_samples)} samples, '
                f'median {median*100:.1f} cm, IQR {(q3-q1)*100:.1f} cm '
                f'(need < {self.arm_length_max_spread*100:.1f} cm)')

    # ---------------------------------------------------------------- calibration
    def current_scale(self, arm):
        return arm.online_scale if arm.online_scale is not None else self.scale_factor

    def map_to_robot(self, arm, p_in_body):
        """Operator position (relative to that arm's shoulder) -> robot target in the output frame.

        Shoulder to shoulder: the position is scaled by robot reach / operator arm length, so the
        operator's full reach maps to the robot's full reach.
        """
        return p_in_body * self.current_scale(arm) + arm.shoulder_offset

    def start_calibration(self):
        self.calibration_required = True
        for arm in self.arms:
            arm.calibrated = False
            arm.cal_samples = []
            arm.online_scale = None
            arm.filtered_wrist_in_body = arm.prev_wrist_in_body = None
            arm.filtered_elbow_in_body = None
        self.get_logger().info('Calibration started: stand relaxed, arms hanging at your sides')

    def calibrate_callback(self, request, response):
        self.start_calibration()
        response.success = True
        response.message = 'Calibration restarted: hold the rest pose still'
        return response

    def update_calibration(self, arm, w_3d, e_3d, w_body, e_body):
        """Collect rest-pose samples; finish once the operator has held still long enough."""
        now = self.get_clock().now().nanoseconds * 1e-9
        shoulder = arm.shoulder_3d
        if arm.cal_samples and now - arm.cal_samples[-1][0] > 0.5:
            arm.cal_samples = []  # lost the operator for a moment: start the hold again
        if shoulder is None or e_3d is None or e_body is None or self.body_landmark_rejected:
            arm.cal_samples = []
            arm.cal_message = 'step back: shoulder, elbow and wrist must all be visible'
            return
        length = np.linalg.norm(e_3d - shoulder) + np.linalg.norm(w_3d - e_3d)
        if not 0.3 < length < 1.2:
            arm.cal_samples = []
            arm.cal_message = 'arm length looks wrong, check depth and lighting'
            return
        # Rest pose = wrist well below the shoulder, roughly under it (not out to the side or front).
        if not (w_body[2] < -0.6 * length and abs(w_body[0]) < 0.35 * length
                and abs(w_body[1]) < 0.35 * length):
            arm.cal_samples = []
            arm.cal_message = f'let your {arm.side} arm hang relaxed at your side'
            return

        arm.cal_samples.append((now, length, w_body.copy(), e_body.copy()))
        wrist = np.array([smp[2] for smp in arm.cal_samples])
        wobble = np.percentile(np.linalg.norm(wrist - np.median(wrist, axis=0), axis=1), 90)
        if wobble > self.cal_still:
            arm.cal_samples = arm.cal_samples[-1:]
            arm.cal_message = 'hold still'
            return
        remaining = self.cal_seconds - (now - arm.cal_samples[0][0])
        if remaining > 0.0 or len(arm.cal_samples) < 15:
            arm.cal_message = f'hold still... {max(remaining, 0.0):.1f} s'
            return

        lengths = np.array([smp[1] for smp in arm.cal_samples])
        q1, median, q3 = np.percentile(lengths, [25, 50, 75])
        if q3 - q1 > self.arm_length_max_spread:
            arm.cal_samples.pop(0)  # depth too noisy to trust yet: keep sampling
            arm.cal_message = 'hold still... (depth is noisy)'
            return
        self.finish_calibration(arm, median)

    def finish_calibration(self, arm, arm_length):
        arm.online_scale = arm.robot_reach / arm_length
        arm.calibrated = True
        arm.cal_samples = []
        arm.filtered_wrist_in_body = arm.prev_wrist_in_body = None
        arm.filtered_elbow_in_body = None
        self.get_logger().info(
            f'Calibrated {arm.side} arm: length {arm_length * 100:.1f} cm -> '
            f'scale {arm.online_scale:.3f}')

    def draw_calibration_status(self, image):
        if not self.calibration_required:
            return
        y = 55
        for arm in self.arms:
            if arm.calibrated:
                text, color = f'{arm.tag}: TRACKING (calibrated)', (0, 255, 0)
            else:
                text, color = f'{arm.tag}: CALIBRATING - {arm.cal_message}', (0, 255, 255)
            cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25
        if not all(arm.calibrated for arm in self.arms):
            cv2.putText(image, 'stand 2-2.5 m away, arms hanging relaxed, hold still', (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def draw_arm_markers(self, image, landmarks, width, height):
        """Overlay each arm's shoulder (cyan), elbow (orange) and the segments between them.

        Drawn from the raw Pose landmarks, so what you see is what the tracker sees,
        independent of whether the depth at that pixel was usable.
        """
        status = f'body axes: {self.body_level}'
        for arm in self.arms:
            points = {}
            for name, index, color in (('S', arm.shoulder_idx, SHOULDER_COLOR),
                                       ('E', arm.elbow_idx, ELBOW_COLOR),
                                       ('W', arm.wrist_idx, self.wrist_color)):
                lm = landmarks[index]
                if lm.visibility > 0.5:
                    points[name] = ((int(lm.x * width), int(lm.y * height)), color)
            for a, b in (('S', 'E'), ('E', 'W')):
                if a in points and b in points:
                    cv2.line(image, points[a][0], points[b][0], (0, 165, 255), 2)
            for name in ('S', 'E'):
                if name in points:
                    (x, y), color = points[name]
                    cv2.circle(image, (x, y), 8, color, -1)
                    cv2.putText(image, arm.tag + name, (x + 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            status += f'   {arm.tag} elbow: {"ok" if "E" in points else "LOST"}'
        cv2.putText(image, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def apply_axis_jump_filter(self, axis_name, new_axis):
        """Apply angular jump filter to limit sudden axis rotations.
        
        If the axis rotates more than axis_jump_threshold, clamp the rotation
        using linear interpolation towards the new direction.
        
        Args:
            axis_name: Identifier for the axis (e.g., 'axis_x')
            new_axis: New unit axis vector (numpy array)
        
        Returns:
            Filtered axis (clamped if angular jump detected)
        """
        if axis_name not in self.prev_axes:
            self.prev_axes[axis_name] = new_axis.copy()
            return new_axis
        
        prev_axis = self.prev_axes[axis_name]
        
        # Calculate angle between axes
        cos_angle = np.clip(np.dot(new_axis, prev_axis), -1.0, 1.0)
        angle = np.arccos(cos_angle)  # in radians
        
        if angle > self.axis_jump_threshold:
            # REJECT the new axis - keep previous (don't interpolate towards bad value!)
            self.get_logger().warn(
                f'AXIS JUMP REJECTED on {axis_name}: {np.degrees(angle):.1f}° (threshold: {np.degrees(self.axis_jump_threshold):.1f}°) - keeping previous'
            )
            # Don't update prev_axes - keep the old good value
            return prev_axis
        else:
            # Accept new axis and update previous
            self.prev_axes[axis_name] = new_axis.copy()
            return new_axis

    def point_3d(self, landmark, depth_image, width, height):
        """3D camera-frame point of a landmark from the depth image, or None if unusable."""
        u, v = int(landmark.x * width), int(landmark.y * height)
        depth = self.get_depth_at_pixel(depth_image, u, v)
        if depth is None or not 0.1 < depth < 10.0:
            return None
        return self.deproject_pixel_to_3d(u, v, depth)

    def update_body_frame(self, landmarks, depth_image, width, height):
        """Body axes from the shoulders (plus hips and ankles for 'up'); per-arm shoulder origins."""
        l_shoulder, r_shoulder = landmarks[11], landmarks[12]
        l_hip, r_hip = landmarks[23], landmarks[24]
        l_ankle, r_ankle = landmarks[27], landmarks[28]
        vis_sh = l_shoulder.visibility > 0.5 and r_shoulder.visibility > 0.5
        vis_hp = l_hip.visibility > 0.5 and r_hip.visibility > 0.5
        vis_ak = l_ankle.visibility > 0.5 and r_ankle.visibility > 0.5
        if not (vis_sh and self.fx is not None):
            return

        def jumped(name, landmark):
            p = self.point_3d(landmark, depth_image, width, height)
            return None if p is None else self.apply_jump_filter(name, p)

        l_sh_3d, r_sh_3d = jumped('l_sh', l_shoulder), jumped('r_sh', r_shoulder)
        if l_sh_3d is None or r_sh_3d is None:
            return
        for arm in self.arms:
            arm.shoulder_3d = r_sh_3d if arm.side == 'right' else l_sh_3d

        up_vec = np.array([0.0, -1.0, 0.0])  # default up (camera frame)
        best_level = 'SHOULDERS_ONLY'
        if vis_hp:
            l_hp_3d, r_hp_3d = jumped('l_hp', l_hip), jumped('r_hp', r_hip)
            if l_hp_3d is not None and r_hp_3d is not None:
                up_vec = (l_sh_3d + r_sh_3d) / 2 - (l_hp_3d + r_hp_3d) / 2
                best_level = 'TORSO'
                if vis_ak:
                    l_ak_3d, r_ak_3d = jumped('l_ak', l_ankle), jumped('r_ak', r_ankle)
                    if l_ak_3d is not None and r_ak_3d is not None:
                        up_vec = (l_sh_3d + r_sh_3d) / 2 - (l_ak_3d + r_ak_3d) / 2
                        best_level = 'FULL_BODY'

        axis_x = (r_sh_3d - l_sh_3d) / (np.linalg.norm(r_sh_3d - l_sh_3d) + 1e-6)
        up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-6)
        axis_z = -np.cross(axis_x, up_vec)
        axis_z /= (np.linalg.norm(axis_z) + 1e-6)
        axis_y = np.cross(axis_x, axis_z)
        self.body_level = best_level

        # Filter and store ONLY if no landmark jumped: keeps the frame consistent.
        if self.body_landmark_rejected:
            if self.frame_count % 30 == 0:
                self.get_logger().warn('Body jump detected - FREEZING axes update for this frame.')
            return
        self.filtered_axis_x = self.apply_ema(axis_x, self.filtered_axis_x, self.alpha_axes)
        self.filtered_axis_y = self.apply_ema(axis_y, self.filtered_axis_y, self.alpha_axes)
        self.filtered_axis_z = self.apply_ema(axis_z, self.filtered_axis_z, self.alpha_axes)
        self.last_valid_R = np.column_stack(
            [self.filtered_axis_z, -self.filtered_axis_x, self.filtered_axis_y])
        # Each arm's origin is its own shoulder. Hips and ankles only help to find "up"; they must
        # not move the origin.
        for arm in self.arms:
            arm.filtered_origin = self.apply_ema(arm.shoulder_3d, arm.filtered_origin, self.alpha_axes)
            arm.last_valid_origin = arm.filtered_origin.copy()
        if self.frame_count % 60 == 0:
            self.get_logger().info(f'Body tracking ACTIVE (Level: {best_level})')

    def process_arm(self, arm, landmarks, depth_image, display_image, width, height):
        """Wrist and elbow of one arm in the body frame (needs a valid body frame)."""
        wrist = landmarks[arm.wrist_idx]
        if wrist.visibility <= 0.5 or arm.last_valid_origin is None or self.last_valid_R is None:
            return
        cv2.circle(display_image, (int(wrist.x * width), int(wrist.y * height)),
                   self.wrist_radius, self.wrist_color, -1)
        w_3d_cam = self.point_3d(wrist, depth_image, width, height)
        if w_3d_cam is None:
            return

        # The elbow is not jump-filtered (that would freeze the body frame on elbow noise); the
        # median depth window and the rule below reject outliers.
        e_3d_cam = None
        elbow = landmarks[arm.elbow_idx]
        if elbow.visibility > 0.5:
            e_3d_cam = self.point_3d(elbow, depth_image, width, height)

        if (self.online_scale_estimation and not self.calibration_required
                and arm.online_scale is None
                and arm.shoulder_3d is not None and e_3d_cam is not None):
            self.update_arm_length(arm, arm.shoulder_3d, e_3d_cam, w_3d_cam)

        w_in_body_raw = self.last_valid_R.T @ (w_3d_cam - arm.last_valid_origin)
        e_in_body_raw = (self.last_valid_R.T @ (e_3d_cam - arm.last_valid_origin)
                         if e_3d_cam is not None else None)
        if self.calibration_required and not arm.calibrated:
            self.update_calibration(arm, w_3d_cam, e_3d_cam, w_in_body_raw, e_in_body_raw)

        # Jump filter: reject jumped values from the EMA entirely
        wrist_jumped = False
        if arm.prev_wrist_in_body is not None:
            wrist_delta = np.linalg.norm(w_in_body_raw - arm.prev_wrist_in_body)
            if wrist_delta >= self.wrist_jump_threshold:
                wrist_jumped = True
                if self.frame_count % 15 == 0:
                    self.get_logger().warn(
                        f'{arm.side} wrist jump rejected: {wrist_delta*100:.1f}cm - keeping filtered')
            else:
                arm.prev_wrist_in_body = w_in_body_raw.copy()
        else:
            arm.prev_wrist_in_body = w_in_body_raw.copy()
        if not wrist_jumped:
            arm.filtered_wrist_in_body = self.apply_ema(
                w_in_body_raw, arm.filtered_wrist_in_body, self.alpha_wrist)

        if e_in_body_raw is not None:
            # Same rejection rule as the wrist: a depth glitch must not move the elbow.
            if (arm.filtered_elbow_in_body is None or
                    np.linalg.norm(e_in_body_raw - arm.filtered_elbow_in_body)
                    < self.wrist_jump_threshold):
                arm.filtered_elbow_in_body = self.apply_ema(
                    e_in_body_raw, arm.filtered_elbow_in_body, self.alpha_wrist)
                arm.elbow_fresh = True

    def make_tf(self, stamp, child, xyz, orientation=None):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self.output_frame
        tf_msg.child_frame_id = child
        (tf_msg.transform.translation.x, tf_msg.transform.translation.y,
         tf_msg.transform.translation.z) = xyz
        q = orientation or Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        tf_msg.transform.rotation.x, tf_msg.transform.rotation.y = q.x, q.y
        tf_msg.transform.rotation.z, tf_msg.transform.rotation.w = q.z, q.w
        return tf_msg

    def publish_arm(self, arm):
        """Publish the last valid wrist (persists through occlusion and jumps). While the arm is
        calibrating, hold the robot at its own rest pose instead."""
        calibrating = self.calibration_required and not arm.calibrated
        if calibrating:
            wrist_final = arm.rest_target.copy()
        elif arm.filtered_wrist_in_body is not None:
            wrist_final = self.map_to_robot(arm, arm.filtered_wrist_in_body)
        else:
            return
        stamp = self.get_clock().now().to_msg()

        if arm.elbow_fresh and not calibrating and arm.filtered_elbow_in_body is not None:
            # Same mapping as the wrist (so the arm keeps its shape); the IK only uses the
            # direction wrist -> elbow from it. Published first so the IK has it for this wrist.
            elbow_final = self.map_to_robot(arm, arm.filtered_elbow_in_body)
            elbow_msg = PoseStamped()
            elbow_msg.header.stamp = stamp
            elbow_msg.header.frame_id = self.output_frame
            (elbow_msg.pose.position.x, elbow_msg.pose.position.y,
             elbow_msg.pose.position.z) = elbow_final
            elbow_msg.pose.orientation.w = 1.0
            arm.elbow_pose_pub.publish(elbow_msg)
            self.tf_broadcaster.sendTransform(
                self.make_tf(stamp, f'{arm.side}_elbow_target', elbow_final))

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.output_frame
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = wrist_final
        pose_msg.pose.orientation = self.hand_quat
        arm.wrist_pose_pub.publish(pose_msg)
        self.tf_broadcaster.sendTransform(
            self.make_tf(stamp, f'{arm.side}_wrist_target', wrist_final, self.hand_quat))

        if self.frame_count % 30 == 0 and not calibrating:
            self.get_logger().info(f'{arm.side} wrist in body: {arm.filtered_wrist_in_body}')

    def synced_callback(self, color_msg, depth_msg):
        """Process synchronized colour and depth images with body frame persistence."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")

            # Handle different depth encodings; the rest of the node works in metres
            if depth_msg.encoding == '32FC1':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            elif depth_msg.encoding == '16UC1':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
                depth_image = depth_image.astype(np.float32) / 1000.0
            else:
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
                if depth_image.dtype == np.uint16:
                    depth_image = depth_image.astype(np.float32) / 1000.0

            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_image)
            display_image = cv_image.copy()
            height, width, _ = cv_image.shape

            self.body_landmark_rejected = False
            for arm in self.arms:
                arm.elbow_fresh = False
                arm.shoulder_3d = None  # never pair a stale shoulder with the current wrist
                if self.calibration_required and not arm.calibrated:
                    arm.cal_message = f'stand in view of the camera, {arm.side} arm visible'

            if results.pose_landmarks:
                self.detection_count += 1
                landmarks = results.pose_landmarks.landmark

                if self.show_all_landmarks:
                    self.mp_drawing.draw_landmarks(
                        display_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

                self.update_body_frame(landmarks, depth_image, width, height)
                self.draw_arm_markers(display_image, landmarks, width, height)
                for arm in self.arms:
                    self.process_arm(arm, landmarks, depth_image, display_image, width, height)

                # Body axes at each shoulder (labelled on the first arm only)
                for n, arm in enumerate(self.arms):
                    if arm.filtered_origin is None or self.filtered_axis_x is None:
                        continue
                    o_px = self.project_3d_to_pixel(arm.filtered_origin)
                    if not o_px:
                        continue
                    for axis, color, label in [(self.filtered_axis_x, (0, 0, 255), 'X'),
                                               (self.filtered_axis_y, (0, 255, 0), 'Y'),
                                               (self.filtered_axis_z, (255, 0, 0), 'Z')]:
                        e_px = self.project_3d_to_pixel(arm.filtered_origin + axis * 0.3)
                        if e_px:
                            cv2.arrowedLine(display_image, o_px, e_px, color, 3)
                            if n == 0:
                                cv2.putText(display_image, label, (e_px[0] + 5, e_px[1]),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            for arm in self.arms:
                self.publish_arm(arm)

            self.frame_count += 1
            self.draw_calibration_status(display_image)
            debug_msg = self.bridge.cv2_to_imgmsg(display_image, 'bgr8')
            debug_msg.header = color_msg.header
            self.debug_image_pub.publish(debug_msg)
            if self.show_window:
                cv2.imshow('Arm Tracking', display_image)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def destroy_node(self):
        """Clean up resources when node is destroyed."""
        self.get_logger().info('Shutting down wrist detector...')
        cv2.destroyAllWindows()
        self.pose.close()
        super().destroy_node()


def main(args=None):
    """Main entry point for the wrist detector node."""
    rclpy.init(args=args)
    node = WristDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
