"""A floating Inspire hand, a table, a red cylinder and a green target disc, in MuJoCo. No arm.

The hand's base link is a free body held by a soft weld to a mocap body: move the mocap body (set_target) and the hand
follows, but it still collides, so it pushes the cylinder and cannot go through the table. Fingers are position servos
driven by the six dex-retargeting actuators (the coupled joints follow through equality constraints, as in the URDF's
<mimic> tags). Same scene numbers as the whole-body backend (src/g1tele/robot/mujoco_backend.py).
"""
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

TABLE_TOP = 0.82
CYL_RADIUS, CYL_HEIGHT = 0.032, 0.12
CYL_START = (0.38, -0.02)
TARGET_XY = (0.38, -0.20)
CONTACT_SOLREF = [0.005, 1.0]                         # stiff contacts: with the default (0.02) the weld pushes the hand 5 cm into the table
MAX_WELD_ERROR = 0.03                                 # m: the weld never pulls from further than this (see step())
FINGER_BANDWIDTH, FINGER_DAMPING = 100.0, 2.4        # servo: kp = I w^2, kd = 2.4 I w (over-damped, like the main backend)


def _patched_urdf(urdf_path):
    """MuJoCo does not read package:// and drops fixed links by default; fix both."""
    path = Path(urdf_path).resolve()
    text = path.read_text(encoding='utf-8').replace('package://dex_hand/urdf/meshes/', '')
    compiler = f'<mujoco><compiler meshdir="{path.parent / "meshes"}" discardvisual="false" fusestatic="false"/></mujoco>'
    text = re.sub(r'(<robot[^>]*>)', lambda m: m.group(1) + compiler, text, count=1)
    out = Path(tempfile.gettempdir()) / f'dexhand_{path.name}'
    out.write_text(text, encoding='utf-8')
    return str(out)


class FloatingHandScene:
    def __init__(self, urdf_path, side, home_position=(0.12, -0.02, 1.02), home_quat=(1, 0, 0, 0)):
        import mujoco
        self.mj = mujoco
        self.side = side
        prefix = 'R' if side == 'right' else 'L'
        self.base_name = f'{prefix}_hand_base_link'
        mimic = {j.get('name'): j.find('mimic') for j in ET.parse(urdf_path).getroot().findall('joint')
                 if j.find('mimic') is not None}

        spec = mujoco.MjSpec.from_file(_patched_urdf(urdf_path))
        spec.option.timestep = 0.001
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.gravity = [0, 0, -9.81]
        for body in spec.bodies:
            body.gravcomp = 1.0                              # the weld and servos need not carry the hand's weight
        for geom in spec.geoms:                              # only the hand's collision meshes (group 0) touch the scene
            if geom.group == 0:
                geom.contype, geom.conaffinity = 2, 0
                geom.friction = [1.5, 0.02, 0.002]
                geom.condim = 4
                geom.solref = CONTACT_SOLREF
            else:
                geom.contype = geom.conaffinity = 0

        base = spec.body(self.base_name)
        base.pos = list(home_position)
        base.quat = list(home_quat)
        base.add_freejoint()
        target = spec.worldbody.add_body(name='target', mocap=True, pos=list(home_position), quat=list(home_quat))
        marker = target.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.008, 0, 0], rgba=[1, 0.8, 0.1, 0.6])
        marker.contype = marker.conaffinity = 0
        weld = spec.add_equality()
        weld.type = mujoco.mjtEq.mjEQ_WELD
        weld.objtype = mujoco.mjtObj.mjOBJ_BODY
        weld.name1, weld.name2 = self.base_name, 'target'
        weld.data[:11] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]   # anchor, relative pose (identity), torque scale
        weld.solref = [0.015, 1.0]

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
                joint.armature = 0.002
                act = spec.add_actuator()
                act.name, act.target = f'act_{joint.name}', joint.name
                act.trntype = mujoco.mjtTrn.mjTRN_JOINT
                act.gaintype, act.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
                act.ctrllimited = 1
                act.ctrlrange = [joint.range[0], joint.range[1]]
                self.actuated.append(joint.name)

        self._add_scene(spec)
        self._tune_servos(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.act = {n: self.model.actuator(f'act_{n}').id for n in self.actuated}
        self.base_id = self.model.body(self.base_name).id
        self.cyl_id = self.model.body('cylinder').id
        self.cyl_qadr = self.model.jnt_qposadr[self.model.body_jntadr[self.cyl_id]]
        self.set_target(home_position, home_quat)
        self.reset_scene()
        self.snap_to_target()

    # -- scene ----------------------------------------------------------------------------------------------
    def _add_scene(self, spec):
        mj = self.mj
        spec.add_texture(name='sky', type=mj.mjtTexture.mjTEXTURE_SKYBOX, builtin=mj.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.30, 0.36, 0.46], rgb2=[0.07, 0.08, 0.11], width=64, height=512)
        light = spec.worldbody.add_light()
        light.pos, light.dir, light.diffuse = [1.5, -1.0, 2.5], [-0.5, 0.35, -0.8], [0.9, 0.9, 0.9]
        spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
        spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
        cam = spec.worldbody.add_camera(name='front', pos=[1.05, -0.80, 1.40], fovy=50)
        cam.mode = mj.mjtCamLight.mjCAMLIGHT_TARGETBODY
        cam.targetbody = 'cylinder'
        table = spec.worldbody.add_geom(type=mj.mjtGeom.mjGEOM_BOX, size=[0.28, 0.55, 0.02],
                                        pos=[0.58, 0.0, TABLE_TOP - 0.02], rgba=[0.55, 0.42, 0.30, 1])
        table.contype, table.conaffinity = 1, 3
        table.friction = [1.0, 0.02, 0.002]
        table.solref = CONTACT_SOLREF
        disc = spec.worldbody.add_geom(type=mj.mjtGeom.mjGEOM_CYLINDER, size=[0.05, 0.0008, 0],
                                       pos=[TARGET_XY[0], TARGET_XY[1], TABLE_TOP + 0.0008], rgba=[0.2, 0.85, 0.3, 1])
        disc.contype = disc.conaffinity = 0
        body = spec.worldbody.add_body(name='cylinder', pos=[*CYL_START, TABLE_TOP + CYL_HEIGHT / 2 + 0.002])
        body.add_freejoint()
        cyl = body.add_geom(type=mj.mjtGeom.mjGEOM_CYLINDER, size=[CYL_RADIUS, CYL_HEIGHT / 2, 0], mass=0.12,
                            rgba=[0.9, 0.25, 0.2, 1])
        cyl.contype, cyl.conaffinity = 1, 3
        cyl.friction = [1.5, 0.02, 0.002]
        cyl.condim = 4
        cyl.solref = CONTACT_SOLREF

    def _tune_servos(self, spec):
        """Gains from each finger joint's inertia, so every finger is over-damped whatever its size."""
        model = spec.compile()
        data = self.mj.MjData(model)
        self.mj.mj_forward(model, data)
        full = np.zeros((model.nv, model.nv))
        self.mj.mj_fullM(model, data, full)
        for act in spec.actuators:
            dof = model.jnt_dofadr[model.joint(act.target).id]
            inertia = max(full[dof, dof], 1e-4)
            w = FINGER_BANDWIDTH
            act.gainprm[0] = inertia * w * w
            act.biasprm[:3] = [0.0, -inertia * w * w, -FINGER_DAMPING * inertia * w]

    # -- control --------------------------------------------------------------------------------------------
    def set_target(self, position, quat_wxyz):
        """Where the hand should be. The hand gets there unless something is in the way."""
        self.goal_pos = np.asarray(position, dtype=float).copy()
        self.data.mocap_pos[0] = position
        self.data.mocap_quat[0] = quat_wxyz

    def set_fingers(self, joints):
        """{joint name: angle}; names without a servo (the coupled joints) are ignored."""
        for name, q in joints.items():
            if name in self.act:
                self.data.ctrl[self.act[name]] = q

    def snap_to_target(self):
        """Put the hand exactly on the target (start-up, or after a big jump) so the weld does not yank it."""
        adr = self.model.jnt_qposadr[self.model.body_jntadr[self.base_id]]
        self.data.mocap_pos[0] = self.goal_pos
        self.data.qpos[adr:adr + 3] = self.goal_pos
        self.data.qpos[adr + 3:adr + 7] = self.data.mocap_quat[0]
        self.data.qvel[:] = 0
        self.mj.mj_forward(self.model, self.data)

    def hand_position(self):
        return self.data.xpos[self.base_id].copy()

    def step(self, n=1):
        """The weld pulls with a force proportional to its error, so a goal far inside the table would overpower the
        contact and push the hand through. The mocap body is therefore kept within MAX_WELD_ERROR of the hand: free
        motion lags by under a centimetre, a blocked hand presses with a bounded force and stays out of the table."""
        for _ in range(n):
            error = self.goal_pos - self.data.xpos[self.base_id]
            norm = np.linalg.norm(error)
            if norm > MAX_WELD_ERROR:
                error *= MAX_WELD_ERROR / norm
            self.data.mocap_pos[0] = self.data.xpos[self.base_id] + error
            self.mj.mj_step(self.model, self.data)

    # -- output ---------------------------------------------------------------------------------------------
    def render(self, width=480, height=360, camera='front'):
        """RGB image of the scene (offscreen). Call it from ONE thread: the GL context lives in the thread that made it."""
        if getattr(self, '_renderer', None) is None:
            self._renderer = self.mj.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def scene_state(self):
        """Plain numbers for the link: where the hand and the cylinder are (positions m, quaternions w x y z)."""
        return {'hand_pos': self.data.xpos[self.base_id].tolist(), 'hand_quat': self.data.xquat[self.base_id].tolist(),
                'cyl_pos': self.data.xpos[self.cyl_id].tolist(), 'cyl_quat': self.data.xquat[self.cyl_id].tolist(),
                **self.status()}

    # -- task -----------------------------------------------------------------------------------------------
    def reset_scene(self):
        a = self.cyl_qadr
        self.data.qpos[a:a + 7] = [*CYL_START, TABLE_TOP + CYL_HEIGHT / 2 + 0.002, 1, 0, 0, 0]
        dof = self.model.body_dofadr[self.cyl_id]
        self.data.qvel[dof:dof + 6] = 0.0
        self.mj.mj_forward(self.model, self.data)

    def status(self):
        p = self.data.xpos[self.cyl_id]
        to_target = float(np.hypot(p[0] - TARGET_XY[0], p[1] - TARGET_XY[1]))
        lifted = max(0.0, float(p[2] - (TABLE_TOP + CYL_HEIGHT / 2)))
        return {'to_target_cm': round(100 * to_target, 1), 'lifted_cm': round(100 * lifted, 1),
                'on_target': bool(to_target < 0.04 and lifted < 0.01)}
