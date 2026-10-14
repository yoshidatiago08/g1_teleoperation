"""Typed configuration, loaded from YAML.

Every tunable lives here with its default and its meaning, so a teammate can copy
`configs/default.yaml`, change a few lines and run. Unknown keys are an error (a typo must not
silently do nothing).
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml

def _find_root() -> str:
    """The folder holding assets/, configs/, profiles/, recordings/: $G1TELE_HOME, else the checkout this
    file lives in, else the current directory (an installed copy run from the repo root)."""
    candidates = [os.environ.get('G1TELE_HOME', ''),
                  os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), os.getcwd()]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, 'assets', 'g1')):
            return os.path.abspath(c)
    return os.path.abspath(candidates[1])


REPO_ROOT = _find_root()


@dataclass
class LinkCfg:
    """Network between the operator's machine (camera) and the host running the robot/simulator.

    The operator machine connects out to the robot host, so only the robot host needs an open
    port. Works over Tailscale, a LAN or a VPN: only `host` changes.
    """
    host: str = '127.0.0.1'      # address of the robot host, as seen from the operator machine
    stream_port: int = 5555      # operator state, notebook -> robot host (latest value only)
    control_port: int = 5556     # reliable events (calibrate, record), notebook -> robot host
    feedback_port: int = 5557    # status and simulator video, robot host -> notebook


@dataclass
class CameraCfg:
    source: str = 'realsense'    # realsense | replay | synthetic
    width: int = 640
    height: int = 480
    fps: int = 30                        # the old pipeline ran 640x480 at 30 fps
    serial: str = ''             # pick one camera if several are plugged in
    depth_preset: str = 'Default'         # RealSense visual preset. 'High Accuracy' drops many pixels on clothing
    temporal_filter: bool = True          # RealSense depth post-processing
    spatial_filter: bool = True
    hole_filling: bool = False            # fills holes with guesses: off for precision
    laser_power: float = -1.0             # emitter power, -1 = keep the camera default
    imu: bool = True                      # use the accelerometer (D435i/D455) to find "up"
    replay_path: str = ''                 # recording to play when source = replay


@dataclass
class JointRadii:
    """Distance from the surface the depth camera sees to the joint centre, in metres.

    The depth image measures the skin in front of the joint, not the joint itself: ignoring this
    puts every joint 2-5 cm too close to the camera.
    """
    shoulder: float = 0.055
    elbow: float = 0.040
    wrist: float = 0.030


@dataclass
class FilterCfg:
    """One Euro filter parameters (min_cutoff in Hz: lower = smoother when still; beta: higher =
    less lag when moving fast). Tune min_cutoff first, then beta."""
    wrist_min_cutoff: float = 1.0
    wrist_beta: float = 5.0
    wrist_depth_min_cutoff: float = 4.0   # Hz: distance to the wrist (noisy) when the arm is seen from the side
    wrist_depth_endon_cutoff: float = 0.35  # Hz: the same when the arm points at the camera (noisier, so smoothed harder)
    wrist_depth_beta: float = 2.0
    elbow_min_cutoff: float = 1.0
    elbow_beta: float = 3.0
    shoulder_min_cutoff: float = 0.8
    shoulder_beta: float = 1.0
    hand_rot_min_cutoff: float = 1.0
    hand_rot_beta: float = 1.0
    finger_min_cutoff: float = 1.5
    finger_beta: float = 3.0
    waist_min_cutoff: float = 1.0
    waist_beta: float = 1.0


@dataclass
class PerceptionCfg:
    arms: List[str] = field(default_factory=lambda: ['right', 'left'])
    hands: bool = True                   # palm orientation and the five fingers
    waist: bool = False                  # waist from the shoulders. Off for manipulation datasets:
                                         # shoulder noise would otherwise move the whole torso
    pose_model_complexity: int = 1       # MediaPipe Pose: 0 lite, 1 full, 2 heavy
    pose_smoothing: bool = True          # MediaPipe's own landmark smoothing (as in the old pipeline)
    fusion: bool = True                  # bone-length depth fusion; false = raw lifted joints, for comparing
    min_visibility: float = 0.5          # Pose landmark visibility needed to trust a joint
    radii: JointRadii = field(default_factory=JointRadii)
    depth_window: int = 9                # pixels, square window around a landmark, for images up to 848 px wide (larger images scale it)
    depth_percentile: float = 35.0       # lower than 50: rejects background bleeding in at limb edges
    depth_range: List[float] = field(default_factory=lambda: [0.25, 4.0])   # metres
    depth_sigma: float = 0.02            # m: error of a good depth reading (noise + radius error). Bigger = trust depth more loosely
    bone_sigma: float = 0.02             # m: how rigid the calibrated bones are. Smaller = bones decide more
    endon_depth_bias: float = 0.10       # m: a hand pointing at the camera is in front of its own wrist, so the wrist depth reads this much too near at full end-on
    hand_crop_margin: float = 2.2
    hand_arm_max_deg: float = 80.0       # the hand's finger axis this far from the forearm is not a hand reading (a flip or the wrong hand)
    hand_pose_max_deg: float = 70.0      # ... or this far from the direction the Pose model gives for the same hand
    hand_max_rate: float = 12.0          # rad/s: the palm cannot turn faster than this (a flip is instant)
    hand_axes_sign: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    finger_autorange: bool = True        # stretch each finger's range to what it actually reaches (never compress), so open is open and a fist is a fist
    finger_autorange_s: float = 20.0     # s of history the stretch looks at
    finger_snap: float = 0.06            # curls within this of 0 or 1 snap to it
    hand_forearm: bool = True            # keep the finger axis inside a cone around the forearm axis (smoothed): the wrist only bends so far
    hand_forearm_weight: float = 0.3     # 0..1: pull of the finger axis toward the forearm axis (0 = only the smoothing and the cone)
    hand_forearm_max_deg: float = 70.0   # the finger axis cannot be further from the forearm than this (wrist flexion + deviation)
    hand_align: bool = True              # hand orientation by aligning the hand model's palm to the depth points (perception/hand_align.py); replaces hand_depth
    hand_align_weight: float = 0.8       # 0..1: how far the palm moves onto the measured palm (scaled by the fit's confidence)
    hand_align_max_deg: float = 60.0     # a correction larger than this is the depth being wrong: ignored, unless the fit is excellent (confidence >= 0.7)
    hand_align_noise_m: float = 0.012    # m: the fit residual of a good alignment; confidence falls beyond it
    hand_depth: bool = True              # palm normal from a plane fitted to the depth image: fixes flipped / edge-on palms
    hand_depth_weight: float = 0.7       # 0..1: how far the palm normal moves to the depth plane when both agree
    hand_depth_flip_deg: float = 50.0    # landmark and depth normals further apart than this: the landmarks are wrong, use the depth
    hand_occlusion: bool = False         # a finger whose landmarks are hidden behind another object keeps its last curl
    occ_front_margin: float = 0.045      # m: a finger landmark this much nearer than the hand should be is hidden by something else
    max_joint_speed: float = 5.0         # m/s: a wrist faster than any human arm is a wrong solution
    shoulder_hold: float = 3.0           # s a hidden shoulder keeps its last position (shoulders barely move)
    shoulder_speed: float = 1.2          # m/s: a shoulder moving faster than this is a bad depth reading
    hold_time: float = 0.25              # s a joint is held after it is lost, before status = LOST
    camera_tilt_deg: float = 0.0         # fallback when neither the IMU nor the calibration gives "up"
    filters: FilterCfg = field(default_factory=FilterCfg)
    send_hz: float = 0.0                 # 0 = send every processed frame


@dataclass
class CalibrationCfg:
    rest_seconds: float = 2.5            # hold the rest pose this long
    auto_recalibrate: bool = True        # re-run the calibration by itself when it detects it has gone wrong
    drift_bone_pct: float = 25.0         # measured arm lengths this far off the calibrated ones for drift_seconds: recalibrate
    drift_seconds: float = 8.0
    drift_up_deg: float = 5.0            # the camera's "up" moved this much since calibration (camera bumped): recalibrate
    stage_timeout: float = 25.0          # s after which an optional pose that cannot be detected is skipped
    pose_seconds: float = 1.5            # hold each extra pose this long
    still_cm: float = 5.0                # wobble allowed while holding
    poses: List[str] = field(default_factory=lambda: ['rest', 'side'])   # 'up' needs ~2.3 m; 'front' (arm at the camera) is unreliable
    hand_poses: bool = True              # open hand, then fist, to learn this person's finger range
    profile_dir: str = os.path.join(REPO_ROOT, 'profiles')   # calibrations are stored here per operator
    operator: str = 'default'


@dataclass
class IKCfg:
    pos_weight: float = 1.0              # wrist position (m)
    elbow_weight: float = 0.8            # elbow position, picks the arm posture (m)
    rot_weight: float = 0.06             # wrist orientation (rad -> m)
    smooth_weight: float = 1e-4          # stay close to the previous solution (kept tiny: it biases tracking)
    posture_weight: float = 1e-5         # faint drift toward a relaxed posture along the null space
    damping: float = 1e-3
    iterations: int = 8
    max_step: float = 0.35               # rad per iteration
    engage_tau: float = 0.8              # s: when an arm (re)engages it starts where it is and eases into your pose
    mapping: str = 'bone_scaled'         # bone_scaled | uniform | metric (see retarget/retargeter.py)
    hand_mode: str = 'curl'              # curl: the six curl values from the finger angles, with your open-hand / fist calibration | vector: fingertip vectors -> Inspire joints (retarget/hand_ik.py, experimental)
    hand_scale: float = 1.0              # vector mode: >1 makes the robot fingers reach further than the operator's (relative to palm size)
    hand_huber: float = 0.02             # m: fingertip error beyond this counts linearly, not squared (a bad landmark cannot dominate)
    hand_reg: float = 0.0003             # stay close to the previous finger angles (smoothness, rad^2 weight)


@dataclass
class RobotCfg:
    urdf: str = os.path.join(REPO_ROOT, 'assets', 'g1', 'g1_29dof_inspire_dfq.urdf')
    backend: str = 'mujoco'              # kinematic | mujoco | dds
    rate_hz: float = 250.0               # backend loop rate
    max_joint_speed: float = 10.0        # rad/s clamp on arm joint targets (safety)
    max_waist_speed: float = 3.0
    max_rot_speed: float = 8.0           # rad/s limit on how fast the wrist orientation target may turn
    lost_rest_after: float = 3.0         # s an arm without data HOLDS its pose before drifting to rest
    rest_speed: float = 0.4              # rad/s of that drift
    watchdog: float = 0.5                # s without operator frames before the robot eases to rest
    render: bool = False                 # mujoco: render the robot and send the video back
    scene: bool = True                   # mujoco: a table, a graspable cylinder and a target disc (press o to reset)
    cameras: List[str] = field(default_factory=lambda: ['head', 'front'])   # mujoco video, side by side: head = first person, front = upper body
    render_size: List[int] = field(default_factory=lambda: [480, 360])   # per camera
    render_fps: float = 60.0
    # dds
    dds_interface: str = ''              # network interface for DDS, e.g. 'lo' or 'eth0'. Empty = default
    dds_domain: int = 1                  # 0 is the real robot's domain; the simulators use 1
    target: str = 'sim'                  # sim | real. 'real' needs --confirm-real on the command line


@dataclass
class RecordCfg:
    enabled: bool = True
    root: str = os.path.join(REPO_ROOT, 'recordings')
    raw_keypoints: bool = True           # store the operator state: lets you re-retarget offline
    fps: float = 30.0


@dataclass
class Config:
    link: LinkCfg = field(default_factory=LinkCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    perception: PerceptionCfg = field(default_factory=PerceptionCfg)
    calibration: CalibrationCfg = field(default_factory=CalibrationCfg)
    ik: IKCfg = field(default_factory=IKCfg)
    robot: RobotCfg = field(default_factory=RobotCfg)
    record: RecordCfg = field(default_factory=RecordCfg)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls, data: Dict[str, Any], path: str = ''):
    """Dataclass from a dict, recursively; unknown keys are rejected."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f'{path or "config"}: expected a mapping, got {type(data).__name__}')
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f'{path or "config"}: unknown key(s) {sorted(unknown)}; '
                         f'valid keys: {sorted(fields)}')
    kwargs = {}
    for name, value in data.items():
        default = getattr(cls(), name)
        if dataclasses.is_dataclass(default):
            kwargs[name] = _build(type(default), value, f'{path}.{name}' if path else name)
        else:
            if isinstance(default, bool) and not isinstance(value, bool):
                raise ValueError(f'{path}.{name}: expected true/false, got {value!r}')
            if isinstance(default, (int, float)) and not isinstance(default, bool):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f'{path}.{name}: expected a number, got {value!r}')
                value = type(default)(value)
            kwargs[name] = value
    return cls(**kwargs)


def _merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | None = None, overrides: Dict[str, Any] | None = None) -> Config:
    """Defaults, then the YAML file, then `overrides` (a nested dict, e.g. from the command line)."""
    data: Dict[str, Any] = {}
    if path:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    if overrides:
        data = _merge(data, overrides)
    return _build(Config, data)


def parse_overrides(items: List[str]) -> Dict[str, Any]:
    """['robot.backend=dds', 'perception.hands=false'] -> nested dict (values parsed as YAML)."""
    out: Dict[str, Any] = {}
    for item in items or []:
        if '=' not in item:
            raise ValueError(f'override {item!r} must look like section.key=value')
        key, value = item.split('=', 1)
        node = out
        parts = key.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


def dump_default_yaml() -> str:
    """configs/default.yaml: every setting with its default and the comment written next to it in
    this file. Regenerate with `python -m g1tele.config > configs/default.yaml`."""
    import inspect
    import re

    def comments_of(cls):
        out = {}
        for line in inspect.getsource(cls).splitlines():
            m = re.match(r'\s{4}(\w+):\s*[^=#]+=.*?#\s*(.*)$', line)
            if m:
                out[m.group(1)] = m.group(2).strip()
        return out

    def render(obj, indent=0):
        pad = '  ' * indent
        lines = []
        notes = comments_of(type(obj))
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            if dataclasses.is_dataclass(value):
                lines.append(f'{pad}{f.name}:')
                lines += render(value, indent + 1)
            else:
                text = yaml.safe_dump(value, default_flow_style=True).strip().replace('\n...', '')
                if text.endswith('\n...'):
                    text = text[:-4]
                note = f'  # {notes[f.name]}' if f.name in notes else ''
                lines.append(f'{pad}{f.name}: {text}{note}')
        return lines

    header = ['# g1tele configuration: every setting with its default. Copy this file, delete what you',
              '# do not change, and pass it with --config. Single values can also be set on the command',
              '# line:  --set robot.backend=dds --set link.host=100.64.0.7', '']
    return '\n'.join(header + render(Config())) + '\n'


if __name__ == '__main__':
    print(dump_default_yaml(), end='')
