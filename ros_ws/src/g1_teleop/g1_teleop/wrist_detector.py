#!/usr/bin/env python3
"""
ROS2 Node for Right Wrist Detection using MediaPipe Pose.

This node subscribes to the ZED camera image and depth topics, detects the right wrist
using MediaPipe Pose (landmark index 16), and visualizes coordinate axes using real depth.

Subscribed Topics:
    /zed/zed_node/rgb/image_rect_color (sensor_msgs/Image): RGB image from ZED camera
    /zed/zed_node/depth/depth_registered (sensor_msgs/Image): Depth image from ZED camera
    /zed/zed_node/depth/camera_info (sensor_msgs/CameraInfo): Camera info from ZED camera

Author: Generated for spot-teleop
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from std_srvs.srv import Trigger
import tf2_ros
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import mediapipe as mp
import numpy as np
import message_filters


class WristDetector(Node):
    """ROS2 node for detecting and visualizing the right wrist using MediaPipe."""
    
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
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Camera info
        self.camera_info = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        # Depth image
        self.depth_image = None
        
        # AprilTag detector setup (36h11 family)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        self.aruco_params = aruco.DetectorParameters()
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Declare parameters
        # [H_G1 port] Defaults switched from ZED to RealSense (aligned depth -> same pixel grid as RGB)
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('show_window', False)
        self.show_window = self.get_parameter('show_window').value
        self.declare_parameter('show_all_landmarks', False)
        self.declare_parameter('wrist_circle_radius', 10)
        self.declare_parameter('wrist_circle_color', [0, 255, 0])  # Green in BGR
        self.declare_parameter('apriltag_size', 0.16)  # AprilTag size in meters (default 10cm)
        
        # Get parameters
        color_topic = self.get_parameter('color_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.show_all_landmarks = self.get_parameter('show_all_landmarks').value
        self.wrist_radius = self.get_parameter('wrist_circle_radius').value
        color_param = self.get_parameter('wrist_circle_color').value
        self.wrist_color = tuple(color_param)
        self.tag_size = self.get_parameter('apriltag_size').value
        
        # Scale factor to map the human wrist displacement to the robot arm.
        # The G1 launch overrides these estimates from the selected URDF.
        # Used at startup and as fallback when online estimation is disabled
        # or has not converged yet.
        self.declare_parameter('scale_factor', 0.41 / 0.65)
        self.scale_factor = self.get_parameter('scale_factor').value

        # Online arm-length estimation: estimate the operator's arm length as
        # the segment sum ||shoulder->elbow|| + ||elbow->wrist||, which is
        # pose-invariant (rigid segments), so no calibration pose is needed.
        self.declare_parameter('online_scale_estimation', True)
        self.online_scale_estimation = self.get_parameter('online_scale_estimation').value
        self.declare_parameter('robot_reach', 0.41)  # Approximate G1 shoulder-to-wrist reach in meters
        self.robot_reach = self.get_parameter('robot_reach').value
        
        # Output frame for wrist pose (robot's body frame)
        self.declare_parameter('output_frame', 'body')
        self.output_frame = self.get_parameter('output_frame').value
        
        # Offset from torso_link to the right shoulder in REP-103 coordinates.
        self.declare_parameter('shoulder_offset', [0.004, -0.100, 0.248])
        self.shoulder_offset = np.array(self.get_parameter('shoulder_offset').value)
        
        # Tracking overlay (skeleton, wrist marker, body axes) as an image topic, so it can be
        # viewed from another machine instead of an OpenCV window.
        self.debug_image_pub = self.create_publisher(Image, '/tracking/debug_image', 1)

        # Elbow position in the same frame and scale as /wrist_pose. The IK uses the direction
        # elbow -> wrist to choose the arm posture; it does not need the elbow's exact position.
        self.elbow_pose_pub = self.create_publisher(PoseStamped, '/elbow_pose', 10)

        # Publisher for wrist pose in body frame
        self.wrist_pose_pub = self.create_publisher(PoseStamped, '/wrist_pose', 10)
        self.create_service(Trigger, 'calibrate', self.calibrate_callback)
        
        # TF broadcaster for wrist target frame
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Create synchronized subscribers for color and depth
        self.color_sub = message_filters.Subscriber(self, Image, color_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        
        # Synchronize color and depth messages
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.synced_callback)
        
        # Camera info subscriber (not synchronized, just need it once)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10
        )
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        
        # EMA filter parameters
        self.declare_parameter('filter_alpha_axes', 0.15)  # Lower = smoother body frame
        self.declare_parameter('filter_alpha_wrist', 0.2)  # Higher = more responsive wrist
        self.alpha_axes = self.get_parameter('filter_alpha_axes').value
        self.alpha_wrist = self.get_parameter('filter_alpha_wrist').value
        
        # Jump filter parameters (max allowed movement per frame in meters)
        self.declare_parameter('jump_threshold', 0.40)  # Increased to 40cm for more fluid body follow
        self.jump_threshold = self.get_parameter('jump_threshold').value
        
        # Wrist jump filter (higher threshold since wrist moves faster)
        self.declare_parameter('wrist_jump_threshold', 0.40)  # 40cm max jump per frame
        self.wrist_jump_threshold = self.get_parameter('wrist_jump_threshold').value
        
        # Angular jump filter (max allowed rotation per frame in degrees)
        self.declare_parameter('axis_jump_threshold_deg', 30.0)  # Increased to 30 degrees
        self.axis_jump_threshold = np.radians(self.get_parameter('axis_jump_threshold_deg').value)
        
        # Filtered states (EMA)
        self.filtered_origin = None
        self.filtered_axis_x = None
        self.filtered_axis_y = None
        self.filtered_axis_z = None
        self.filtered_wrist_in_body = None
        self.filtered_elbow_in_body = None
        self.elbow_fresh = False  # True only if the elbow was measured in the current frame

        # --- Calibration: the operator holds a relaxed rest pose (arm hanging at the side) so
        # the arm length can be measured cleanly. See the README.
        self.declare_parameter('calibration_required', True)
        self.declare_parameter('calibration_seconds', 3.0)   # how long to hold still
        self.declare_parameter('calibration_still_cm', 6.0)  # allowed wrist wobble while holding
        self.declare_parameter('rest_target', [0.0, 0.0, 0.0])  # robot wrist at rest (output frame)
        self.calibration_required = self.get_parameter('calibration_required').value
        self.cal_seconds = float(self.get_parameter('calibration_seconds').value)
        self.cal_still = float(self.get_parameter('calibration_still_cm').value) / 100.0
        self.rest_target = np.array(self.get_parameter('rest_target').value, dtype=float)
        self.calibrated = not self.calibration_required
        self.cal_samples = []        # (time, arm length, wrist in body, elbow in body)
        self.cal_message = ''
        self.body_level = 'NONE'  # which landmarks gave the body axes (shown on the overlay)
        
        # Previous landmark positions for jump filter
        self.prev_landmarks_3d = {}
        
        # Previous wrist_in_body for jump filter
        self.prev_wrist_in_body = None

        # Online arm-length estimation state
        self.arm_length_samples = []       # rolling window of segment-sum lengths (m)
        self.arm_length_window = 150       # ~5 s at 30 Hz
        self.arm_length_min_samples = 90   # samples required before latching
        self.arm_length_max_spread = 0.03  # IQR convergence threshold (m)
        self.online_scale = None           # latched scale; None until converged
        self.current_r_sh_3d = None        # right shoulder 3D of the current frame
        
        # Previous axes for angular jump filter
        self.prev_axes = {}
        
        # Max plausible velocity for body landmarks (m/s)
        # Shoulders/hips don't move faster than ~1.5 m/s in normal use
        self.declare_parameter('max_landmark_velocity', 1.5)
        self.max_landmark_velocity = self.get_parameter('max_landmark_velocity').value
        
        # Timestamps for velocity-based convergence
        self.prev_landmark_times = {}
        
        # Last valid body frame (used when any landmark is rejected)
        self.last_valid_origin = None
        self.last_valid_R = None  # Rotation matrix for body frame
        
        # Flag to track if any body landmark was rejected this frame
        self.body_landmark_rejected = False
        
        # Hand orientation (roll)
        self.hand_quat = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.hand_roll_sub = self.create_subscription(
            Quaternion,
            '/hand_roll_quat',
            self.hand_roll_callback,
            10
        )
        
        self.get_logger().info('=== Wrist Detector Node Started ===')
        self.get_logger().info(f'Subscribing to color topic: {color_topic}')
        self.get_logger().info(f'Subscribing to depth topic: {depth_topic}')
        self.get_logger().info(f'Subscribing to camera info: {camera_info_topic}')
        self.get_logger().info(f'Right wrist landmark index: 16 (MediaPipe Pose)')
        self.get_logger().info(f'Show all landmarks: {self.show_all_landmarks}')
        self.get_logger().info(f'AprilTag size: {self.tag_size} meters')
        self.get_logger().info(f'Output frame for wrist pose: {self.output_frame}')
        self.get_logger().info(f'Filter alpha (axes): {self.alpha_axes}, (wrist): {self.alpha_wrist}')
        self.get_logger().info(f'Jump threshold: {self.jump_threshold*100:.1f} cm/frame')
        self.get_logger().info(f'Max landmark velocity: {self.max_landmark_velocity:.2f} m/s')
        self.get_logger().info(f'Wrist jump threshold: {self.wrist_jump_threshold*100:.1f} cm/frame')
        self.get_logger().info(f'Axis jump threshold: {np.degrees(self.axis_jump_threshold):.1f} deg/frame')
        self.get_logger().info(f'Scale factor (human to G1 arm): {self.scale_factor:.3f}')
        self.get_logger().info(
            f'Online scale estimation: {self.online_scale_estimation} '
            f'(robot reach: {self.robot_reach:.3f} m)')
        self.get_logger().info(f'Shoulder offset (body->sh0): X={self.shoulder_offset[0]:.3f}, Y={self.shoulder_offset[1]:.3f}, Z={self.shoulder_offset[2]:.3f}')
        
        # Store last comparison results for logging
        self.last_position_error = None
        self.last_angle_errors = None

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

    def update_arm_length(self, sh_3d, el_3d, wr_3d):
        """Accumulate arm-length samples and latch the scale on convergence.

        The upper-arm and forearm segment lengths are pose-invariant, so
        ||elbow - shoulder|| + ||wrist - elbow|| estimates the operator's
        arm length at any flexion, with no calibration pose. The median of
        a rolling window rejects depth outliers; once enough samples agree,
        the scale is latched so the hand-to-robot mapping does not keep
        drifting during operation.

        Args:
            sh_3d: Right shoulder 3D position, camera frame (numpy array)
            el_3d: Right elbow 3D position, camera frame (numpy array)
            wr_3d: Right wrist 3D position, camera frame (numpy array)
        """
        length = np.linalg.norm(el_3d - sh_3d) + np.linalg.norm(wr_3d - el_3d)
        if not (0.3 < length < 1.2):
            return  # implausible arm length — depth artifact

        self.arm_length_samples.append(length)
        if len(self.arm_length_samples) > self.arm_length_window:
            self.arm_length_samples.pop(0)

        if len(self.arm_length_samples) < self.arm_length_min_samples:
            return

        q1, median, q3 = np.percentile(self.arm_length_samples, [25, 50, 75])
        if (q3 - q1) < self.arm_length_max_spread:
            self.online_scale = self.robot_reach / median
            self.get_logger().info(
                f'Arm length converged: {median*100:.1f} cm '
                f'(IQR {(q3-q1)*100:.1f} cm, {len(self.arm_length_samples)} samples) '
                f'-> scale latched at {self.online_scale:.3f}')
        elif self.frame_count % 150 == 0:
            self.get_logger().info(
                f'Estimating arm length: {len(self.arm_length_samples)} samples, '
                f'median {median*100:.1f} cm, IQR {(q3-q1)*100:.1f} cm '
                f'(need < {self.arm_length_max_spread*100:.1f} cm)')

    # ---------------------------------------------------------------- calibration
    def current_scale(self):
        return self.online_scale if self.online_scale is not None else self.scale_factor

    def map_to_robot(self, p_in_body):
        """Operator position (relative to the right shoulder) -> robot target in the output frame.

        Shoulder to shoulder: the position is scaled by robot reach / operator arm length, so the
        operator's full reach maps to the robot's full reach.
        """
        return p_in_body * self.current_scale() + self.shoulder_offset

    def start_calibration(self):
        self.calibration_required = True
        self.calibrated = False
        self.cal_samples = []
        self.online_scale = None
        self.filtered_wrist_in_body = self.prev_wrist_in_body = None
        self.filtered_elbow_in_body = None
        self.get_logger().info('Calibration started: stand relaxed, right arm hanging at your side')

    def calibrate_callback(self, request, response):
        self.start_calibration()
        response.success = True
        response.message = 'Calibration restarted: hold the rest pose still'
        return response

    def update_calibration(self, w_3d, e_3d, w_body, e_body):
        """Collect rest-pose samples; finish once the operator has held still long enough."""
        now = self.get_clock().now().nanoseconds * 1e-9
        shoulder = self.current_r_sh_3d
        if self.cal_samples and now - self.cal_samples[-1][0] > 0.5:
            self.cal_samples = []  # lost the operator for a moment: start the hold again
        if shoulder is None or e_3d is None or e_body is None or self.body_landmark_rejected:
            self.cal_samples = []
            self.cal_message = 'step back: shoulder, elbow and wrist must all be visible'
            return
        length = np.linalg.norm(e_3d - shoulder) + np.linalg.norm(w_3d - e_3d)
        if not 0.3 < length < 1.2:
            self.cal_samples = []
            self.cal_message = 'arm length looks wrong, check depth and lighting'
            return
        # Rest pose = wrist well below the shoulder, roughly under it (not out to the side or front).
        if not (w_body[2] < -0.6 * length and abs(w_body[0]) < 0.35 * length
                and abs(w_body[1]) < 0.35 * length):
            self.cal_samples = []
            self.cal_message = 'let your right arm hang relaxed at your side'
            return

        self.cal_samples.append((now, length, w_body.copy(), e_body.copy()))
        wrist = np.array([smp[2] for smp in self.cal_samples])
        wobble = np.percentile(np.linalg.norm(wrist - np.median(wrist, axis=0), axis=1), 90)
        if wobble > self.cal_still:
            self.cal_samples = self.cal_samples[-1:]
            self.cal_message = 'hold still'
            return
        remaining = self.cal_seconds - (now - self.cal_samples[0][0])
        if remaining > 0.0 or len(self.cal_samples) < 15:
            self.cal_message = f'hold still... {max(remaining, 0.0):.1f} s'
            return

        lengths = np.array([smp[1] for smp in self.cal_samples])
        q1, median, q3 = np.percentile(lengths, [25, 50, 75])
        if q3 - q1 > self.arm_length_max_spread:
            self.cal_samples.pop(0)  # depth too noisy to trust yet: keep sampling
            self.cal_message = 'hold still... (depth is noisy)'
            return
        self.finish_calibration(median)

    def finish_calibration(self, arm_length):
        self.online_scale = self.robot_reach / arm_length
        self.calibrated = True
        self.cal_samples = []
        self.filtered_wrist_in_body = self.prev_wrist_in_body = None
        self.filtered_elbow_in_body = None
        self.get_logger().info(
            f'Calibrated: arm length {arm_length * 100:.1f} cm -> scale {self.online_scale:.3f}')

    def draw_calibration_status(self, image):
        if self.calibration_required and not self.calibrated:
            cv2.putText(image, 'CALIBRATING: ' + self.cal_message, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(image, 'stand 2-2.5 m away, right arm hanging relaxed, hold still', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        elif self.calibration_required:
            cv2.putText(image, 'TRACKING (calibrated)', (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def draw_arm_markers(self, image, landmarks, width, height):
        """Overlay the right shoulder (cyan), elbow (orange) and the arm between them.

        Drawn from the raw Pose landmarks, so what you see is what the tracker sees,
        independent of whether the depth at that pixel was usable.
        """
        pose = self.mp_pose.PoseLandmark
        points = {}
        for name, index, color in (('S', pose.RIGHT_SHOULDER, (255, 255, 0)),
                                   ('E', pose.RIGHT_ELBOW, (0, 165, 255)),
                                   ('W', pose.RIGHT_WRIST, (0, 255, 0))):
            lm = landmarks[index.value]
            if lm.visibility > 0.5:
                points[name] = ((int(lm.x * width), int(lm.y * height)), color)
        for a, b in (('S', 'E'), ('E', 'W')):
            if a in points and b in points:
                cv2.line(image, points[a][0], points[b][0], (0, 165, 255), 2)
        for name in ('S', 'E'):
            if name in points:
                (x, y), color = points[name]
                cv2.circle(image, (x, y), 8, color, -1)
                cv2.putText(image, name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        status = f'body axes: {self.body_level}   elbow: {"ok" if "E" in points else "LOST"}'
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

    def synced_callback(self, color_msg, depth_msg):
        """Process synchronized color and depth images with body frame persistence."""
        try:
            # Convert ROS Images to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            
            # Handle different depth encodings
            if depth_msg.encoding == '32FC1':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            elif depth_msg.encoding == '16UC1':
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
                depth_image = depth_image.astype(np.float32) / 1000.0
            else:
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
                if depth_image.dtype == np.uint16:
                    depth_image = depth_image.astype(np.float32) / 1000.0
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_image)
            display_image = cv_image.copy()
            height, width, _ = cv_image.shape
            
            # Reset rejection flag for this frame
            self.body_landmark_rejected = False
            self.elbow_fresh = False
            if self.calibration_required and not self.calibrated:
                self.cal_message = 'stand in view of the camera, right arm visible'
            
            if results.pose_landmarks:
                self.detection_count += 1
                landmarks = results.pose_landmarks.landmark

                # Shoulder of the current frame only (for arm-length estimation);
                # never pair a stale shoulder with the current wrist
                self.current_r_sh_3d = None

                # Draw landmarks if requested
                if self.show_all_landmarks:
                    self.mp_drawing.draw_landmarks(
                        display_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                
                # 1. Update Body Frame with Hierarchical Fallback
                l_shoulder, r_shoulder = landmarks[11], landmarks[12]
                l_hip, r_hip = landmarks[23], landmarks[24]
                l_ankle, r_ankle = landmarks[27], landmarks[28]
                
                # Check visibility for different levels
                vis_sh = l_shoulder.visibility > 0.5 and r_shoulder.visibility > 0.5
                vis_hp = l_hip.visibility > 0.5 and r_hip.visibility > 0.5
                vis_ak = l_ankle.visibility > 0.5 and r_ankle.visibility > 0.5
                
                if vis_sh and self.fx is not None:
                    # Basic points for all levels
                    sh_l_px = (int(l_shoulder.x * width), int(l_shoulder.y * height))
                    sh_r_px = (int(r_shoulder.x * width), int(r_shoulder.y * height))
                    d_sh_l = self.get_depth_at_pixel(depth_image, sh_l_px[0], sh_l_px[1])
                    d_sh_r = self.get_depth_at_pixel(depth_image, sh_r_px[0], sh_r_px[1])
                    
                    if d_sh_l and d_sh_r:
                        l_sh_3d = self.apply_jump_filter('l_sh', self.deproject_pixel_to_3d(sh_l_px[0], sh_l_px[1], d_sh_l))
                        r_sh_3d = self.apply_jump_filter('r_sh', self.deproject_pixel_to_3d(sh_r_px[0], sh_r_px[1], d_sh_r))
                        self.current_r_sh_3d = r_sh_3d

                        # Initialize vectors
                        up_vec = np.array([0.0, -1.0, 0.0]) # Default up (camera frame)
                        
                        best_level = "None"
                        
                        # --- LEVEL 1: Full Body (Ankles) ---
                        if vis_hp and vis_ak:
                            hp_l_px = (int(l_hip.x * width), int(l_hip.y * height))
                            hp_r_px = (int(r_hip.x * width), int(r_hip.y * height))
                            ak_l_px = (int(l_ankle.x * width), int(l_ankle.y * height))
                            ak_r_px = (int(r_ankle.x * width), int(r_ankle.y * height))
                            d_hp_l = self.get_depth_at_pixel(depth_image, hp_l_px[0], hp_l_px[1])
                            d_hp_r = self.get_depth_at_pixel(depth_image, hp_r_px[0], hp_r_px[1])
                            d_ak_l = self.get_depth_at_pixel(depth_image, ak_l_px[0], ak_l_px[1])
                            d_ak_r = self.get_depth_at_pixel(depth_image, ak_r_px[0], ak_r_px[1])
                            
                            if all([d_hp_l, d_hp_r, d_ak_l, d_ak_r]):
                                l_hp_3d = self.apply_jump_filter('l_hp', self.deproject_pixel_to_3d(hp_l_px[0], hp_l_px[1], d_hp_l))
                                r_hp_3d = self.apply_jump_filter('r_hp', self.deproject_pixel_to_3d(hp_r_px[0], hp_r_px[1], d_hp_r))
                                l_ak_3d = self.apply_jump_filter('l_ak', self.deproject_pixel_to_3d(ak_l_px[0], ak_l_px[1], d_ak_l))
                                r_ak_3d = self.apply_jump_filter('r_ak', self.deproject_pixel_to_3d(ak_r_px[0], ak_r_px[1], d_ak_r))
                                
                                up_vec = (l_sh_3d + r_sh_3d)/2 - (l_ak_3d + r_ak_3d)/2
                                best_level = "FULL_BODY"
                        
                        # --- LEVEL 2: Torso Only (Hips) ---
                        if best_level == "None" and vis_hp:
                            hp_l_px = (int(l_hip.x * width), int(l_hip.y * height))
                            hp_r_px = (int(r_hip.x * width), int(r_hip.y * height))
                            d_hp_l = self.get_depth_at_pixel(depth_image, hp_l_px[0], hp_l_px[1])
                            d_hp_r = self.get_depth_at_pixel(depth_image, hp_r_px[0], hp_r_px[1])
                            
                            if d_hp_l and d_hp_r:
                                l_hp_3d = self.apply_jump_filter('l_hp', self.deproject_pixel_to_3d(hp_l_px[0], hp_l_px[1], d_hp_l))
                                r_hp_3d = self.apply_jump_filter('r_hp', self.deproject_pixel_to_3d(hp_r_px[0], hp_r_px[1], d_hp_r))
                                up_vec = (l_sh_3d + r_sh_3d)/2 - (l_hp_3d + r_hp_3d)/2
                                best_level = "TORSO"
                                
                        # --- LEVEL 3: Shoulders Only ---
                        if best_level == "None":
                            best_level = "SHOULDERS_ONLY"
                            # up_vec is already set to camera default [0, -1, 0]
                        
                        # Common Frame Calculation
                        axis_x = (r_sh_3d - l_sh_3d) / (np.linalg.norm(r_sh_3d - l_sh_3d) + 1e-6)
                        up_vec /= (np.linalg.norm(up_vec) + 1e-6)
                        axis_z = -np.cross(axis_x, up_vec)
                        axis_z /= (np.linalg.norm(axis_z) + 1e-6)
                        axis_y = np.cross(axis_x, axis_z)
                        
                        # The origin is always the right shoulder. Hips and ankles only help to
                        # find which way is "up"; they must not move the origin (it used to sit at
                        # the torso centre, which jumped ~25 cm whenever they entered or left view).
                        origin_3d = r_sh_3d.copy()
                        self.body_level = best_level
                        
                        # Filter and store ONLY if no landmark jumped
                        # This keeps the axis frame consistent (no weird twists from partial updates)
                        if not self.body_landmark_rejected:
                            self.filtered_origin = self.apply_ema(origin_3d, self.filtered_origin, self.alpha_axes)
                            self.filtered_axis_x = self.apply_ema(axis_x, self.filtered_axis_x, self.alpha_axes)
                            self.filtered_axis_y = self.apply_ema(axis_y, self.filtered_axis_y, self.alpha_axes)
                            self.filtered_axis_z = self.apply_ema(axis_z, self.filtered_axis_z, self.alpha_axes)
                            
                            self.last_valid_origin = self.filtered_origin.copy()
                            self.last_valid_R = np.column_stack([self.filtered_axis_z, -self.filtered_axis_x, self.filtered_axis_y])
                            
                            if self.frame_count % 60 == 0:
                                self.get_logger().info(f'Body tracking ACTIVE (Level: {best_level})')
                        else:
                            if self.frame_count % 30 == 0:
                                self.get_logger().warn('Body jump detected - FREEZING axes update for this frame.')
                
                self.draw_arm_markers(display_image, landmarks, width, height)

                # 2. Process Wrist independently using last valid body frame
                right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                if right_wrist.visibility > 0.5 and self.last_valid_origin is not None:
                    w_x, w_y = int(right_wrist.x * width), int(right_wrist.y * height)
                    cv2.circle(display_image, (w_x, w_y), self.wrist_radius, self.wrist_color, -1)
                    
                    w_depth = self.get_depth_at_pixel(depth_image, w_x, w_y)
                    if w_depth is not None and 0.1 < w_depth < 10.0:
                        w_3d_cam = self.deproject_pixel_to_3d(w_x, w_y, w_depth)
                        if w_3d_cam is not None:
                            # Online arm-length estimation: needs shoulder, elbow,
                            # and wrist from the same frame. The elbow is not
                            # jump-filtered (that would freeze the body frame on
                            # elbow noise); the median window rejects outliers.
                            e_3d_cam = None
                            r_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                            if r_elbow.visibility > 0.5:
                                e_x, e_y = int(r_elbow.x * width), int(r_elbow.y * height)
                                e_depth = self.get_depth_at_pixel(depth_image, e_x, e_y)
                                if e_depth is not None and 0.1 < e_depth < 10.0:
                                    e_3d_cam = self.deproject_pixel_to_3d(e_x, e_y, e_depth)

                            if (self.online_scale_estimation and not self.calibration_required
                                    and self.online_scale is None
                                    and self.current_r_sh_3d is not None and e_3d_cam is not None):
                                self.update_arm_length(self.current_r_sh_3d, e_3d_cam, w_3d_cam)

                            w_in_body_raw = self.last_valid_R.T @ (w_3d_cam - self.last_valid_origin)
                            e_in_body_raw = (self.last_valid_R.T @ (e_3d_cam - self.last_valid_origin)
                                             if e_3d_cam is not None else None)
                            if self.calibration_required and not self.calibrated:
                                self.update_calibration(w_3d_cam, e_3d_cam, w_in_body_raw, e_in_body_raw)
                            
                            # Jump filter: reject jumped values from EMA entirely
                            wrist_jumped = False
                            if self.prev_wrist_in_body is not None:
                                wrist_delta = np.linalg.norm(w_in_body_raw - self.prev_wrist_in_body)
                                if wrist_delta >= self.wrist_jump_threshold:
                                    wrist_jumped = True
                                    if self.frame_count % 15 == 0:
                                        self.get_logger().warn(
                                            f'Wrist jump rejected: {wrist_delta*100:.1f}cm — keeping filtered')
                                else:
                                    self.prev_wrist_in_body = w_in_body_raw.copy()
                            else:
                                self.prev_wrist_in_body = w_in_body_raw.copy()
                            
                            if not wrist_jumped:
                                self.filtered_wrist_in_body = self.apply_ema(
                                    w_in_body_raw, self.filtered_wrist_in_body, self.alpha_wrist)

                            if e_in_body_raw is not None:
                                # Same rejection rule as the wrist: a depth glitch must not move the elbow.
                                if (self.filtered_elbow_in_body is None or
                                        np.linalg.norm(e_in_body_raw - self.filtered_elbow_in_body)
                                        < self.wrist_jump_threshold):
                                    self.filtered_elbow_in_body = self.apply_ema(
                                        e_in_body_raw, self.filtered_elbow_in_body, self.alpha_wrist)
                                    self.elbow_fresh = True
                
                # 3. Visualization and Logging
                if self.filtered_origin is not None:
                    o_px = self.project_3d_to_pixel(self.filtered_origin)
                    if o_px:
                        for axis, color, label in [(self.filtered_axis_x, (0,0,255), 'X'), (self.filtered_axis_y, (0,255,0), 'Y'), (self.filtered_axis_z, (255,0,0), 'Z')]:
                            e_px = self.project_3d_to_pixel(self.filtered_origin + axis * 0.3)
                            if e_px: 
                                cv2.arrowedLine(display_image, o_px, e_px, color, 3)
                                cv2.putText(display_image, label, (e_px[0]+5, e_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
            # Publish the last valid wrist (persists through occlusion and jumps). While
            # calibrating, hold the robot at its own rest pose instead.
            calibrating = self.calibration_required and not self.calibrated
            if calibrating:
                wrist_final = self.rest_target.copy()
            elif self.filtered_wrist_in_body is not None:
                wrist_final = self.map_to_robot(self.filtered_wrist_in_body)
            else:
                wrist_final = None

            if wrist_final is not None:
                stamp = self.get_clock().now().to_msg()

                if (not calibrating and self.elbow_fresh and self.filtered_elbow_in_body is not None
                        and self.filtered_wrist_in_body is not None):
                    # Same mapping as the wrist (so the arm keeps its shape); the IK only uses the
                    # direction wrist -> elbow from it.
                    elbow_final = self.map_to_robot(self.filtered_elbow_in_body)
                    elbow_msg = PoseStamped()
                    elbow_msg.header.stamp = stamp
                    elbow_msg.header.frame_id = self.output_frame
                    (elbow_msg.pose.position.x, elbow_msg.pose.position.y,
                     elbow_msg.pose.position.z) = elbow_final
                    elbow_msg.pose.orientation.w = 1.0
                    self.elbow_pose_pub.publish(elbow_msg)
                    elbow_tf = TransformStamped()
                    elbow_tf.header = elbow_msg.header
                    elbow_tf.child_frame_id = "elbow_target"
                    (elbow_tf.transform.translation.x, elbow_tf.transform.translation.y,
                     elbow_tf.transform.translation.z) = elbow_final
                    elbow_tf.transform.rotation.w = 1.0
                    self.tf_broadcaster.sendTransform(elbow_tf)

                pose_msg = PoseStamped()
                pose_msg.header.stamp = stamp
                pose_msg.header.frame_id = self.output_frame
                pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = wrist_final
                pose_msg.pose.orientation = self.hand_quat
                self.wrist_pose_pub.publish(pose_msg)

                tf_msg = TransformStamped()
                tf_msg.header.stamp = stamp
                tf_msg.header.frame_id = self.output_frame
                tf_msg.child_frame_id = "wrist_target"
                tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z = wrist_final
                tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z, tf_msg.transform.rotation.w = \
                    self.hand_quat.x, self.hand_quat.y, self.hand_quat.z, self.hand_quat.w
                self.tf_broadcaster.sendTransform(tf_msg)

                if self.frame_count % 30 == 0 and not calibrating:
                    self.get_logger().info(f'Wrist in Body: {self.filtered_wrist_in_body}')
            
            self.frame_count += 1
            self.draw_calibration_status(display_image)
            debug_msg = self.bridge.cv2_to_imgmsg(display_image, 'bgr8')
            debug_msg.header = color_msg.header
            self.debug_image_pub.publish(debug_msg)
            if self.show_window:
                cv2.imshow('Right Wrist Detection', display_image)
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
