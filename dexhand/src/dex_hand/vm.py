"""VM side: everything heavy, no ROS and no display.

    frames in ─► MediaPipe ─► dex-retargeting + scaled motion ─► MuJoCo (physics, offscreen render) ─► state + JPEGs out
    events in ─► clutch / scale / reset

    python -m dex_hand.vm --hand right                  # waits for the notebook's frames
    python -m dex_hand.vm --source synthetic            # a made-up hand, no notebook needed (events and state still work)

Two threads: the main one follows the camera rate (MediaPipe is the slow step); the physics one runs MuJoCo in real
time, applies the newest hand command and renders for the notebook, so the contact physics never waits for MediaPipe.
"""
import argparse
import os
import threading
import time

os.environ.setdefault('MUJOCO_GL', 'egl')       # offscreen rendering on the VM's GPU (osmesa = software, no GPU)

import cv2
import numpy as np

from dex_hand import capture, frames, link, paths
from dex_hand.motion import MotionScaler
from dex_hand.mujoco_scene import FloatingHandScene
from dex_hand.pipeline import HandPipeline
from dex_hand.retarget import FingerRetargeter
from dex_hand.sources import Observation, wrist_depth

# Wrist above the table, far enough behind the cylinder (edge at x 0.35) that even curled fingers (reach about 0.19 m) clear it:
# a hand that starts overlapping the cylinder launches it.
HOME = (0.12, -0.02, 1.02)
HOME_ROTATION = np.diag([1.0, -1.0, -1.0])   # fingers forward, palm down
FRAME_TIMEOUT = 0.5                     # s without a frame = the hand is lost (the robot holds)
SNAP_DISTANCE = 0.25                    # m: a first target this far from the hand teleports it there
SIM_DT = 0.001
CAL_SECONDS = 1.5                       # how long the calibration pose is measured
CAL_MIN_SAMPLES = 15
CAL_MAX_SPREAD = np.radians(12)         # the hand must be this still while calibrating
MAX_ROTATION_RATE = 6.0                 # rad/s the robot hand may turn (re-engaging swings it there smoothly)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12), (13, 14),
         (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]


def wxyz(rotation):
    x, y, z, w = frames.quaternion(rotation)
    return [w, x, y, z]


class VmApp:
    def __init__(self, side='right', source='frames', motion_scale=0.3, rotation_scale=1.0, finger_scale=1.15,
                 port=link.DEFAULT_PORT, render=True, render_size=(480, 360), render_hz=20.0):
        self.side, self.render, self.render_size, self.render_hz = side, render, render_size, render_hz
        self.retargeter = FingerRetargeter(paths.URDF_DIR, paths.retarget_config(side), side, finger_scale)
        self.scaler = MotionScaler(HOME, HOME_ROTATION, motion_scale, rotation_scale, max_rotation_rate=MAX_ROTATION_RATE)
        self.pipeline = HandPipeline(side, self.retargeter, self.scaler, 600.0, 320.0, 240.0)
        home_quat = wxyz(HOME_ROTATION @ self.retargeter.r_bh.T)          # the simulated hand starts at home, palm down
        self.scene = FloatingHandScene(paths.urdf(side), side, home_position=HOME, home_quat=home_quat)
        self.detector = self.synthetic = None
        if source == 'synthetic':
            from dex_hand.synthetic import SyntheticHandSource
            self.synthetic = SyntheticHandSource(side)
        else:
            from dex_hand.sources import HandDetector
            self.detector = HandDetector(side)
        self.frames_in, self.state_out, self.events_in = link.vm_side(port)
        self.lock = threading.Lock()
        # (position, quaternion wxyz, joints) the simulated hand follows. Starts at home with open fingers and only changes while
        # imitating, so before the calibration, and while paused, nothing moves.
        self.goal = (np.array(HOME), home_quat, {name: 0.0 for name in self.retargeter.joint_names})
        self.reset_requested = False
        self.sim_state = self.scene.scene_state()
        self.sim_jpeg = None
        self.paused = False                           # the operator's pause key
        self.calibrated = False                       # imitation starts only after the first successful calibration
        self.camera_up = None                         # from the camera's IMU, when it has one
        self.calibrating = None                       # {'end': monotonic time, 'samples': [...]} while measuring
        self.calibration = {'state': 'none'}          # what the notebook shows

    # -- events ---------------------------------------------------------------------------------------------
    def _handle_events(self):
        while True:
            event = self.events_in.recv(0)
            if event is None:
                return
            cmd = event.get('cmd')
            if cmd == 'clutch':
                self.paused = not self.paused
            elif cmd == 'scale':
                self.scaler.set_scale(scale=self.scaler.scale * float(event['factor']))
            elif cmd == 'set_scale':
                self.scaler.set_scale(scale=event.get('motion'), rotation_scale=event.get('rotation'))
            elif cmd == 'reset':
                with self.lock:
                    self.reset_requested = True
            elif cmd == 'calibrate':
                self.calibrating = {'end': time.monotonic() + CAL_SECONDS, 'samples': []}
                self.calibration = {'state': 'collecting'}

    def _calibrate_step(self, obs, cmd):
        """While calibrating the robot holds (see run) and your hand's rotation is collected; at the end the calibration is
        installed if the hand was tracked and still."""
        if self.calibrating is None:
            return
        if cmd is not None and obs.world is not None:
            self.calibrating['samples'].append(cmd.raw_rotation)
        if time.monotonic() < self.calibrating['end']:
            return
        samples, self.calibrating = self.calibrating['samples'], None
        if len(samples) < CAL_MIN_SAMPLES:
            self.calibration = {'state': 'failed', 'why': 'hand not seen: show the palm to the camera, fingers up'}
            return
        rotation, spread = frames.calibration_from(samples)
        if spread > CAL_MAX_SPREAD:
            self.calibration = {'state': 'failed', 'why': f'hand moved ({np.degrees(spread):.0f} deg): hold still'}
            return
        self._install_calibration(rotation, spread)

    # -- tracking -------------------------------------------------------------------------------------------
    def _install_calibration(self, rotation, spread):
        self.pipeline.set_calibration(rotation)
        self.calibration = {'state': 'done', 'spread_deg': round(float(np.degrees(spread)), 1)}
        self.calibrated, self.paused = True, False                    # calibrated: start imitating

    @property
    def imitating(self):
        return self.calibrated and not self.paused and self.calibrating is None

    def _status(self, tracking):
        if self.calibrating is not None:
            return 'CALIBRATING'
        if not self.calibrated:
            return 'PRESS C TO CALIBRATE'
        if self.paused:
            return 'PAUSED'
        return 'IMITATING' if tracking else 'HAND LOST (holding)'

    def _observe(self, message):
        bgr, depth = capture.decode(message)
        self.pipeline.focal_px, self.pipeline.cx, self.pipeline.cy = message['fx'], message['cx'], message['cy']
        if message.get('up') is not None:
            self.camera_up = np.asarray(message['up'], dtype=float)
            self.pipeline.set_camera_up(self.camera_up)
        found = self.detector(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if found is None:
            return Observation(None, None, None, bgr)
        world, pixels = found
        d = None
        if depth is not None:
            d = wrist_depth(depth, pixels[0], message['depth_scale'], message['depth_downscale'])
        return Observation(world, pixels, d, bgr)

    def _preview(self, obs, tracking):
        if obs.image is None:
            return None
        image = obs.image.copy()
        if obs.pixels is not None:
            for a, b in EDGES:
                cv2.line(image, tuple(obs.pixels[a].astype(int)), tuple(obs.pixels[b].astype(int)), (0, 220, 0), 2)
        status = self._status(tracking)
        cv2.putText(image, f'{status}   scale {self.scaler.scale:.2f}', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
        scale = 480.0 / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale)
        return cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()

    def _camera_state(self):
        if self.camera_up is None:
            return {'imu': False}
        u = self.camera_up                           # pitch: the optical axis above the horizontal; roll: about it
        return {'imu': True, 'pitch_deg': round(float(np.degrees(np.arcsin(np.clip(u[2], -1, 1)))), 1),
                'roll_deg': round(float(np.degrees(np.arcsin(np.clip(u[0], -1, 1)))), 1)}

    def _publish(self, obs, cmd, t_capture):
        state = {'t': time.time(), 't_capture': t_capture, 'tracking': obs.world is not None, 'clutch': self.paused,
                 'status': self._status(obs.world is not None), 'imitating': self.imitating,
                 'scale': self.scaler.scale, 'rotation_scale': self.scaler.rotation_scale, 'calibration': self.calibration,
                 'camera': self._camera_state()}
        with self.lock:
            goal_pos, goal_quat, goal_joints = self.goal                    # what the simulated hand actually follows
        state.update(joints=goal_joints, goal_pos=goal_pos.tolist(), goal_quat=list(goal_quat), engaged=False)
        if cmd is not None:
            state.update(engaged=cmd.engaged, depth_from=cmd.depth_from,
                         ghost=None if cmd.ghost is None else {'pos': cmd.ghost[0].tolist(),
                                                               'quat': wxyz(cmd.ghost[1])},
                         skeleton=cmd.skeleton.tolist())
        with self.lock:
            state['scene'], sim = self.sim_state, self.sim_jpeg
        self.state_out.send({'state': state, 'preview': self._preview(obs, state['tracking']), 'sim': sim})

    # -- loops ----------------------------------------------------------------------------------------------
    def run(self, stop):
        physics = threading.Thread(target=self._physics, args=(stop,), daemon=True)
        physics.start()
        last_frame = time.monotonic()
        while not stop.is_set():
            self._handle_events()
            t_capture = None
            if self.synthetic is not None:
                obs = self.synthetic.read()
            else:
                message = self.frames_in.recv(timeout_ms=20)
                if message is not None:
                    obs, t_capture, last_frame = self._observe(message), message['t'], time.monotonic()
                elif time.monotonic() - last_frame > FRAME_TIMEOUT:
                    obs = Observation(None, None, None, None)       # no frames: the hand is lost, the robot holds
                else:
                    continue
            cmd = self.pipeline.step(obs, not self.imitating)
            self._calibrate_step(obs, cmd)
            if cmd is not None and cmd.engaged:                              # only while imitating and the hand is seen
                with self.lock:
                    self.goal = (cmd.position, wxyz(cmd.rotation), cmd.joints)
            self._publish(obs, cmd, t_capture)
        physics.join(timeout=2.0)

    def _physics(self, stop):
        scene, snapped = self.scene, False
        clock, next_render = time.monotonic(), 0.0
        while not stop.is_set():
            with self.lock:
                goal, reset, self.reset_requested = self.goal, self.reset_requested, False
            if reset:
                scene.reset_scene()
            if goal is not None:
                position, quat, joints = goal
                scene.set_target(position, quat)
                scene.set_fingers(joints)
                if not snapped or np.linalg.norm(position - scene.hand_position()) > SNAP_DISTANCE:
                    scene.snap_to_target()
                    snapped = True
            now = time.monotonic()
            n = min(int((now - clock) / SIM_DT), 100)               # real time, however long rendering took
            if n:
                scene.step(n)
                clock = clock + n * SIM_DT if n < 100 else now
            else:
                time.sleep(0.0005)
                continue
            jpeg = None
            if self.render and now >= next_render:
                rgb = scene.render(*self.render_size)
                jpeg = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
                next_render = now + 1.0 / self.render_hz
            state = scene.scene_state()
            with self.lock:
                self.sim_state = state
                if jpeg is not None:
                    self.sim_jpeg = jpeg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hand', default='right', choices=['right', 'left'])
    parser.add_argument('--source', default='frames', choices=['frames', 'synthetic'])
    parser.add_argument('--motion-scale', type=float, default=0.3)
    parser.add_argument('--rotation-scale', type=float, default=1.0)
    parser.add_argument('--finger-scale', type=float, default=1.15)
    parser.add_argument('--port', type=int, default=link.DEFAULT_PORT)
    parser.add_argument('--no-render', action='store_true', help='physics only: no simulator video for the notebook')
    args = parser.parse_args(argv)
    app = VmApp(args.hand, args.source, args.motion_scale, args.rotation_scale, args.finger_scale, args.port,
                render=not args.no_render)
    print(f'dexhand VM: {args.hand} hand, source {args.source}, ports {args.port}-{args.port + 2}', flush=True)
    stop = threading.Event()
    try:
        app.run(stop)
    except KeyboardInterrupt:
        stop.set()


if __name__ == '__main__':
    main()
