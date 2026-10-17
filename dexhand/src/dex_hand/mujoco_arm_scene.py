"""The G1 (waist and legs static, left arm relaxed, Inspire hands) at the table with the red cylinder, in MuJoCo. The right arm is moved by
joint position servos (from the IK), the right hand's fingers by the dex-retargeting angles.

Same scene numbers, servo tuning and hand collisions as the main project's MuJoCo backend (src/g1tele/robot/mujoco_backend.py), without
its legs. Only the hands collide with the scene; the IK keeps the arm clear of the table, and the cylinder is for the hand to pick up.
"""
import numpy as np

from dex_hand import paths
from dex_hand.mujoco_scene import CONTACT_SOLREF, FINGER_MAX_TORQUE, TaskScene, _patched_urdf

PELVIS_Z = 0.8
ARM_JOINTS = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint',
              'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
LEFT_ARM_RELAXED = {'left_shoulder_pitch_joint': -0.2094, 'left_shoulder_roll_joint': 0.3316, 'left_elbow_joint': 0.5236}
# Servo bandwidth (rad/s) and damping factor by joint family; kp = I w^2, kd = factor I w (over-damped), as in the main backend
BANDWIDTH = {'shoulder': 110.0, 'elbow': 110.0, 'wrist': 110.0, 'waist': 40.0, 'finger': 100.0}
DAMPING = {'shoulder': 3.2, 'elbow': 3.2, 'wrist': 2.4, 'waist': 2.4, 'finger': 2.4}


def _family(name):
    if name.startswith(('R_', 'L_')):
        return 'finger'
    for key in ('shoulder', 'elbow', 'wrist', 'waist'):
        if key in name:
            return key
    return 'waist'


class ArmScene(TaskScene):
    def __init__(self, urdf_path=None, arm_start=None, finger_torque=FINGER_MAX_TORQUE):
        import mujoco
        import xml.etree.ElementTree as ET
        self.mj = mujoco
        urdf_path = urdf_path or paths.G1_URDF
        mimic = {j.get('name'): j.find('mimic') for j in ET.parse(urdf_path).getroot().findall('joint') if j.find('mimic') is not None}

        spec = mujoco.MjSpec.from_file(_patched_urdf(urdf_path))
        spec.option.timestep = 0.001
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.gravity = [0, 0, -9.81]
        spec.body('pelvis').pos = [0.0, 0.0, PELVIS_Z]
        for body in spec.bodies:
            body.gravcomp = 1.0                              # the servos need not carry the arm's weight: exact tracking
        for geom in spec.geoms:                              # only the hands (collision meshes, group 0) touch the scene
            body = geom.parent.name if getattr(geom, 'parent', None) is not None else ''
            if geom.group == 0 and (body.startswith(('R_', 'L_')) or body.endswith('wrist_yaw_link')):
                geom.contype, geom.conaffinity = 2, 0
                geom.friction = [1.5, 0.02, 0.002]
                geom.condim = 4
                geom.solref = CONTACT_SOLREF
            else:
                geom.contype = geom.conaffinity = 0

        self.actuated = []
        for joint in spec.joints:
            if joint.name in mimic:
                m = mimic[joint.name]
                eq = spec.add_equality()
                eq.type = mujoco.mjtEq.mjEQ_JOINT
                eq.name1, eq.name2 = joint.name, m.get('joint')
                eq.data[:5] = [float(m.get('offset', 0)), float(m.get('multiplier', 1)), 0, 0, 0]
                eq.solref = [0.004, 1.0]
            elif joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                joint.armature = 0.002 if _family(joint.name) == 'finger' else 0.02
                act = spec.add_actuator()
                act.name, act.target = f'act_{joint.name}', joint.name
                act.trntype = mujoco.mjtTrn.mjTRN_JOINT
                act.gaintype, act.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
                act.ctrllimited = 1
                act.ctrlrange = [joint.range[0], joint.range[1]]
                if finger_torque and _family(joint.name) == 'finger':          # the arm keeps its full torque; only the fingers are limited
                    act.forcelimited = 1
                    act.forcerange = [-finger_torque, finger_torque]
                self.actuated.append(joint.name)

        self._add_scene(spec)
        self._tune_servos(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.act = {n: self.model.actuator(f'act_{n}').id for n in self.actuated}
        self.qadr = {self.model.joint(i).name: self.model.jnt_qposadr[i] for i in range(self.model.njnt)
                     if self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE}
        self.base_id = self.model.body('R_hand_base_link').id
        self.cyl_id = self.model.body('cylinder').id
        self.cyl_qadr = self.model.jnt_qposadr[self.model.body_jntadr[self.cyl_id]]
        self.set_arm_pose({**LEFT_ARM_RELAXED, **(arm_start or {})})
        self.reset_scene()

    def _tune_servos(self, spec):
        """Gains from each joint's inertia at a mid-reach pose, so every joint is over-damped whatever its size."""
        model = spec.compile()
        data = self.mj.MjData(model)
        for side in ('left', 'right'):
            for joint, value in ((f'{side}_shoulder_pitch_joint', -0.9), (f'{side}_elbow_joint', 0.9)):
                data.qpos[model.jnt_qposadr[model.joint(joint).id]] = value
        self.mj.mj_forward(model, data)
        full = np.zeros((model.nv, model.nv))
        self.mj.mj_fullM(model, data, full)
        for act in spec.actuators:
            dof = model.jnt_dofadr[model.joint(act.target).id]
            inertia = max(full[dof, dof], 1e-4)
            family = _family(act.target)
            w = BANDWIDTH[family]
            act.gainprm[0] = inertia * w * w
            act.biasprm[:3] = [0.0, -inertia * w * w, -DAMPING[family] * inertia * w]

    # -- control --------------------------------------------------------------------------------------------
    def set_arm_pose(self, joints):
        """Put joints at these angles at once (qpos and servo target): the start pose, not a motion."""
        for name, q in joints.items():
            if name in self.qadr:
                self.data.qpos[self.qadr[name]] = q
            if name in self.act:
                self.data.ctrl[self.act[name]] = q
        self.data.qvel[:] = 0.0
        self.mj.mj_forward(self.model, self.data)

    def set_arm_targets(self, joints):
        """Servo targets for the right arm: {joint name: angle}; other names are ignored."""
        for name in ARM_JOINTS:
            if name in joints:
                self.data.ctrl[self.act[name]] = joints[name]

    def set_fingers(self, joints):
        """{joint name: angle}; names without a servo (the coupled joints) are ignored."""
        for name, q in joints.items():
            if name in self.act and name.startswith('R_'):
                self.data.ctrl[self.act[name]] = q

    def apply_goal(self, goal):
        """goal: {'arm': {right-arm joint: angle} or None, 'fingers': {joint: angle}} (the same call as FloatingHandScene's)."""
        if goal.get('arm'):
            self.set_arm_targets(goal['arm'])
        self.set_fingers(goal['fingers'])

    def scene_state(self):
        return {**super().scene_state(), 'robot_joints': self.robot_joints()}

    def step(self, n=1):
        for _ in range(n):
            self.mj.mj_step(self.model, self.data)

    def robot_joints(self):
        """Every joint angle {name: rad} (the coupled finger joints included), for RViz."""
        return {name: float(self.data.qpos[adr]) for name, adr in self.qadr.items() if not name.startswith('cylinder')}
