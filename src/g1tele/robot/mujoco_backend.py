"""MuJoCo backend: the G1 (fixed at the pelvis) with Inspire hands, arms and fingers driven by
position servos. Built straight from the URDF the retargeter uses, so both agree on every joint.

Meant for fast development and for datasets where you only need the arms/hands in a simple scene;
Isaac Sim is reached through the DDS backend.
"""
from __future__ import annotations

import os
import threading
from typing import Dict, Optional

import numpy as np

from g1tele.config import REPO_ROOT, Config
from g1tele.retarget.retargeter import RobotTargets
from g1tele.retarget.urdf import Urdf, arm_joint_names, hand_joint_names
from g1tele.robot.base import RobotBackend, curl_to_joint

CACHE = os.path.join(REPO_ROOT, '.cache', 'mujoco')

# Damping factor: kd = factor * I * w. Shoulders swing the whole arm, so their inertia during a motion is
# larger than at the pose it was measured in: they get extra damping to stay free of overshoot.
DAMPING = {'shoulder': 3.2, 'elbow': 3.2, 'wrist': 2.4, 'waist': 2.4, 'leg': 2.4, 'finger': 2.4}
# Servo bandwidth (rad/s) by joint family. Gains are derived per joint from its inertia so that every
# joint is over-damped: kp = I w^2. Tracking lag of a moving target is about kd/kp = 25-30 ms.
BANDWIDTH = {'shoulder': 110.0, 'elbow': 110.0, 'wrist': 110.0, 'waist': 40.0, 'leg': 40.0, 'finger': 100.0}


def _family(name: str) -> str:
    if name.startswith(('R_', 'L_')):
        return 'finger'
    for key in ('shoulder', 'elbow', 'wrist', 'waist'):
        if key in name:
            return key
    return 'leg'


TABLE_TOP = 0.82                 # m
CYL_RADIUS, CYL_HEIGHT = 0.032, 0.12
CYL_START = (0.38, -0.02)         # x, y on the table, in front of the robot
TARGET_XY = (0.38, -0.20)         # the green disc to bring the cylinder to


def build_spec(urdf_path: str, collisions: bool = False, scene: bool = True):  # noqa: C901
    """MjSpec of the robot: URDF + servos on every joint + mimic constraints + a floor and cameras."""
    import mujoco
    os.makedirs(CACHE, exist_ok=True)
    src_dir = os.path.dirname(urdf_path)
    text = open(urdf_path, encoding='utf-8').read().replace('package://g1_teleop/urdf/meshes/', '')
    # MuJoCo's default is to fuse fixed links into their parent, and it DROPS the fixed joint's
    # rotation/offset for the fused link's children: the Inspire hands came out rotated (right hand
    # 180 degrees, left 90) and offset. Keep fixed links as bodies instead.
    text = text.replace('discardvisual="false"', 'discardvisual="false" fusestatic="false"')
    patched = os.path.join(CACHE, os.path.basename(urdf_path))
    open(patched, 'w', encoding='utf-8').write(text)
    link = os.path.join(CACHE, 'meshes')
    if not os.path.exists(link):
        os.symlink(os.path.join(src_dir, 'meshes'), link)

    spec = mujoco.MjSpec.from_file(patched)
    spec.option.timestep = 0.001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST   # stable with stiff servos on light links
    spec.option.gravity = [0, 0, -9.81]
    urdf = Urdf(urdf_path)

    # Float the pelvis at standing height; arms and hands are what we drive.
    pelvis = spec.body('pelvis')
    pelvis.pos = [0, 0, 0.8]
    for body in spec.bodies:
        body.gravcomp = 1.0          # the servos need not fight gravity: exact tracking
    for geom in spec.geoms:
        # The robot does not collide with itself. Only the hands (collision meshes, group 0) collide with
        # the scene: bit 2 = hand, bit 1 = scene (see the scene below).
        body = geom.parent.name if getattr(geom, 'parent', None) is not None else ''
        is_hand = body.startswith(('R_', 'L_')) or body.endswith('wrist_yaw_link')
        if scene and is_hand and geom.group == 0:
            geom.contype, geom.conaffinity = 2, 0
            geom.friction = [1.5, 0.02, 0.002]
            geom.condim = 4
        elif not collisions:
            geom.contype = 0
            geom.conaffinity = 0

    mimic = {n: j.mimic for n, j in urdf.joints.items() if j.mimic is not None}
    for joint in spec.joints:
        if joint.name in mimic:
            master, mult, offset = mimic[joint.name]
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_JOINT
            eq.name1, eq.name2 = joint.name, master
            eq.data[:5] = [offset, mult, 0, 0, 0]
            eq.solref = [0.004, 1.0]
            continue
        kp, kv = 100.0, 5.0        # placeholders: retuned from the joint inertia in MujocoBackend
        joint.armature = 0.002 if _family(joint.name) == 'finger' else 0.02
        act = spec.add_actuator()
        act.name = f'act_{joint.name}'
        act.target = joint.name
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = kp
        act.biasprm[:3] = [0.0, -kp, -kv]
        act.ctrllimited = 1
        act.ctrlrange = [joint.range[0], joint.range[1]]

    # Manipulation view: the legs and the floor are hidden (they only get in the way of seeing the
    # arms and hands), the background is a dark gradient, and the camera looks from the front-right
    # so reaching forward and to the sides both stay visible.
    for geom in spec.geoms:
        body = geom.parent.name if getattr(geom, 'parent', None) is not None else ''
        if any(k in body for k in ('hip', 'knee', 'ankle')):
            geom.rgba = [0, 0, 0, 0]
    tex = spec.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.30, 0.36, 0.46],
                           rgb2=[0.07, 0.08, 0.11], width=64, height=512)
    light = spec.worldbody.add_light()
    light.pos = [1.5, -1.0, 2.5]
    light.dir = [-0.5, 0.35, -0.8]
    light.diffuse = [0.9, 0.9, 0.9]
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    cam = spec.worldbody.add_camera()
    cam.name = 'front'
    cam.pos = [1.25, -0.8, 1.25]
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    cam.targetbody = 'torso_link'
    cam.fovy = 58
    # first-person camera on the robot's head (fixed to the torso): looks forward and down at the table
    tilt = np.radians(48)
    head = spec.body('torso_link').add_camera()
    head.name = 'head'
    head.pos = [0.09, 0.0, 0.47]
    head.mode = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    head.fovy = 80
    # camera x = robot's right, y = up in the image, looking along -z_cam (forward and down)
    from scipy.spatial.transform import Rotation
    x_axis, y_axis = np.array([0, -1.0, 0]), np.array([np.sin(tilt), 0, np.cos(tilt)])
    q = Rotation.from_matrix(np.column_stack([x_axis, y_axis, np.cross(x_axis, y_axis)])).as_quat()
    head.quat = [q[3], q[0], q[1], q[2]]
    if scene:
        table = spec.worldbody.add_geom()
        table.type = mujoco.mjtGeom.mjGEOM_BOX
        table.size = [0.28, 0.55, 0.02]
        table.pos = [0.58, 0.0, TABLE_TOP - 0.02]
        table.rgba = [0.55, 0.42, 0.30, 1]
        table.contype, table.conaffinity = 1, 3
        table.friction = [1.0, 0.02, 0.002]
        disc = spec.worldbody.add_geom()               # the target: visual only
        disc.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        disc.size = [0.05, 0.0008, 0]
        disc.pos = [TARGET_XY[0], TARGET_XY[1], TABLE_TOP + 0.0008]
        disc.rgba = [0.2, 0.85, 0.3, 1]
        disc.contype = disc.conaffinity = 0
        body = spec.worldbody.add_body()
        body.name = 'cylinder'
        body.pos = [CYL_START[0], CYL_START[1], TABLE_TOP + CYL_HEIGHT / 2 + 0.002]
        body.add_freejoint()
        cyl = body.add_geom()
        cyl.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        cyl.size = [CYL_RADIUS, CYL_HEIGHT / 2, 0]
        cyl.mass = 0.12
        cyl.rgba = [0.9, 0.25, 0.2, 1]
        cyl.contype, cyl.conaffinity = 1, 3
        cyl.friction = [1.5, 0.02, 0.002]
        cyl.condim = 4
    return spec


class MujocoBackend(RobotBackend):
    def __init__(self, cfg: Config):
        import mujoco
        self.mj = mujoco
        self.cfg = cfg
        self.urdf = Urdf(cfg.robot.urdf)
        spec = build_spec(cfg.robot.urdf, scene=cfg.robot.scene)
        self._tune_servos(spec)
        self.model = spec.compile()
        self.data = self.mj.MjData(self.model)
        self.rate_hz = cfg.robot.rate_hz
        self.substeps = max(1, round(1.0 / (self.rate_hz * self.model.opt.timestep)))
        self.lock = threading.Lock()
        self.act_of: Dict[str, int] = {}
        for i in range(self.model.nu):
            name = self.model.actuator(i).name
            self.act_of[name[len('act_'):]] = i
        self.qadr = {self.model.joint(i).name: self.model.jnt_qposadr[i] for i in range(self.model.njnt)}
        self.sides = list(cfg.perception.arms)
        self.hand_act = {s: [self.act_of[n] for n in hand_joint_names(s)] for s in self.sides
                         if all(n in self.act_of for n in hand_joint_names(s))}
        self.targets = RobotTargets()
        self.renderer = None
        self.viewer = None
        # hold every joint at its current position until targets arrive
        self.data.ctrl[:] = 0.0
        self.cyl = self.model.body('cylinder').id if cfg.robot.scene else None
        self._cyl_qpos = self.model.jnt_qposadr[self.model.body_jntadr[self.cyl]] if self.cyl is not None else None
        self.mj.mj_forward(self.model, self.data)

    def _tune_servos(self, spec):
        """Critically damped position servos: gains from each joint's inertia at the zero pose."""
        mj = self.mj
        model = spec.compile()
        data = mj.MjData(model)
        # inertia at a mid-reach pose (arms raised, elbows bent): at the zero pose the arms hang
        # close to the shoulder and the inertia is much smaller than during manipulation
        for side in ('left', 'right'):
            for joint, value in ((f'{side}_shoulder_pitch_joint', -0.9), (f'{side}_elbow_joint', 0.9)):
                data.qpos[model.jnt_qposadr[model.joint(joint).id]] = value
        mj.mj_forward(model, data)
        full = np.zeros((model.nv, model.nv))
        mj.mj_fullM(model, data, full)
        for act in spec.actuators:
            joint = act.target
            dof = model.jnt_dofadr[model.joint(joint).id]
            inertia = max(full[dof, dof], 1e-4)
            w = BANDWIDTH[_family(joint)]
            act.gainprm[0] = inertia * w * w
            damping = DAMPING[_family(joint)]
            act.biasprm[:3] = [0.0, -inertia * w * w, -damping * inertia * w]

    def initialize(self, targets: RobotTargets) -> None:
        with self.lock:
            for name, q in {**targets.arm_q, **targets.waist_q}.items():
                if name in self.qadr:
                    self.data.qpos[self.qadr[name]] = q
            self.mj.mj_forward(self.model, self.data)
        self.apply(targets)

    def apply(self, targets: RobotTargets) -> None:
        with self.lock:
            for name, q in {**targets.arm_q, **targets.waist_q}.items():
                if name in self.act_of:
                    self.data.ctrl[self.act_of[name]] = q
            for side, curls in targets.hand.items():
                if side in self.hand_act:
                    for act, q in zip(self.hand_act[side], curl_to_joint(curls)):
                        self.data.ctrl[act] = q
            self.targets = targets

    def step(self) -> None:
        with self.lock:
            for _ in range(self.substeps):
                self.mj.mj_step(self.model, self.data)

    def observe(self) -> Dict[str, float]:
        with self.lock:
            return {n: float(self.data.qpos[a]) for n, a in self.qadr.items()}

    def reset_scene(self) -> None:
        if self._cyl_qpos is None:
            return
        with self.lock:
            a = self._cyl_qpos
            self.data.qpos[a:a + 7] = [CYL_START[0], CYL_START[1], TABLE_TOP + CYL_HEIGHT / 2 + 0.002, 1, 0, 0, 0]
            self.data.qvel[self.model.body_dofadr[self.cyl]:self.model.body_dofadr[self.cyl] + 6] = 0.0

    def scene_status(self) -> Dict[str, float]:
        if self.cyl is None:
            return {}
        with self.lock:
            p = self.data.xpos[self.cyl].copy()
        to_target = float(np.hypot(p[0] - TARGET_XY[0], p[1] - TARGET_XY[1]))
        lifted = max(0.0, float(p[2] - (TABLE_TOP + CYL_HEIGHT / 2)))
        return {'to_target_cm': round(100 * to_target, 1), 'lifted_cm': round(100 * lifted, 1),
                'on_target': bool(to_target < 0.04 and lifted < 0.01)}

    def render(self) -> Optional[np.ndarray]:
        """The configured cameras side by side. Call it from ONE thread (the GL context lives there); the
        physics lock is held only while the scene is copied, not while drawing, so rendering never stalls
        the control loop."""
        w, h = self.cfg.robot.render_size
        if self.renderer is None:
            os.environ.setdefault('MUJOCO_GL', 'egl')
            self.renderer = self.mj.Renderer(self.model, height=h, width=w)
        tiles = []
        for cam in self.cfg.robot.cameras:
            with self.lock:
                self.renderer.update_scene(self.data, camera=cam)
            tiles.append(self.renderer.render().copy())
        return tiles[0] if len(tiles) == 1 else np.hstack(tiles)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
