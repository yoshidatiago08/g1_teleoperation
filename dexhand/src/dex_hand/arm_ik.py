"""Right-arm inverse kinematics on the GPU with CuRobo (v0.8, "cuRoboV2"): the hand target in, 7 joint angles out.

The robot is the G1 with the waist and everything below locked; the tool frame is R_hand_base_link, the same frame dex-retargeting and
the floating-hand simulation use, so the scaled hand target goes in unchanged. The table is an obstacle (the IK keeps the arm and hand
spheres out of it); the cylinder is deliberately NOT, it is what you pick up.

Two solvers: a full multi-seed IK (to start at the home pose, and as a fallback when tracking keeps failing), and a one-seed IK warm-started from
the previous solution, which stays in the same arm configuration and takes about 3 ms. (CuRobo's "differential IK" mode, run_optimizer=False,
does not move the arm in the v0.8.0 release: the LM seed solver only runs together with the optimizer.) The joint change per frame is limited,
so the arm never jumps. Positions are in the world frame (z up, the pelvis at PELVIS_Z); CuRobo works in the pelvis frame.
"""
import time

import numpy as np
import yaml

from dex_hand import paths
from dex_hand.mujoco_scene import TABLE_TOP

PELVIS_Z = 0.8                      # the pelvis height in the simulation (as in the main project's MuJoCo scene)
ARM_JOINTS = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint',
              'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
TABLE_CENTER, TABLE_DIMS = (0.58, 0.0, TABLE_TOP - 0.02), (0.56, 1.10, 0.04)


def robot_config():
    """The generated CuRobo robot config with this machine's absolute asset paths filled in."""
    cfg = yaml.safe_load(open(paths.CUROBO_CONFIG))
    cfg['robot_cfg']['kinematics']['urdf_path'] = str(paths.G1_URDF)
    cfg['robot_cfg']['kinematics']['asset_root_path'] = str(paths.G1_MESHES)
    return cfg


def table_scene():
    x, y, z = TABLE_CENTER
    return {'cuboid': {'table': {'dims': list(TABLE_DIMS), 'pose': [x, y, z - PELVIS_Z, 1.0, 0.0, 0.0, 0.0]}}}


class ArmIK:
    def __init__(self, avoid_table=True, rate_hz=30.0, max_joint_speed=3.0, num_seeds=32):
        import torch
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        self.torch = torch
        self.max_step = max_joint_speed / rate_hz            # rad per frame
        robot, scene = robot_config(), (table_scene() if avoid_table else None)
        self.full = InverseKinematics(InverseKinematicsCfg.create(
            robot=robot, scene_model=scene, num_seeds=num_seeds, self_collision_check=True, collision_cache={'cuboid': 4}))
        self.tracker = InverseKinematics(InverseKinematicsCfg.create(
            robot=robot, scene_model=scene, optimizer_configs=['ik/lbfgs_ik.yml'], num_seeds=1, seed_solver_num_seeds=1,
            use_cuda_graph=True, position_tolerance=0.002, orientation_tolerance=0.02, self_collision_check=True))
        self.frame = self.tracker.tool_frames[0]
        self.joint_names = list(self.full.default_joint_state.joint_names)       # the 7 active (right arm) joints
        self.state = None                                    # the last solution (a CuRobo JointState): the next solve starts from it
        self.last_error_mm = 0.0
        self.failures = 0                                    # consecutive frames the one-seed solve did not reach the target

    def _goal(self, position_world, quat_wxyz):
        from curobo.types import GoalToolPose, Pose
        p = np.asarray(position_world, dtype=np.float32) - np.array([0.0, 0.0, PELVIS_Z], dtype=np.float32)
        pose = Pose(position=self.torch.tensor(p[None], device='cuda'),
                    quaternion=self.torch.tensor(np.asarray(quat_wxyz, dtype=np.float32)[None], device='cuda'))
        return GoalToolPose.from_poses({self.frame: pose}, ordered_tool_frames=self.tracker.tool_frames, num_goalset=1)

    def start(self, position_world, quat_wxyz, steps=60):
        """Put the arm at the start pose. Begins at the relaxed default pose and walks the target there in small steps, so the arm ends in a
        natural configuration (a many-seed solve gives odd wrist angles that then collide with the body). Returns {joint: angle}, or None if
        the pose cannot be reached."""
        default = {n: float(v) for n, v in zip(self.joint_names, self.full.default_joint_state.position.reshape(-1).tolist())}
        p0, q0 = self.forward(default)
        goal = self._goal(p0, q0)
        result = self.tracker.solve_pose(goal, current_state=self.full.default_joint_state.clone().unsqueeze(0), return_seeds=1)
        self.state = result.js_solution.clone()
        self.failures = 0
        step, self.max_step = self.max_step, 1.0              # no speed limit while positioning (this is not the operator moving)
        try:
            for s in np.linspace(0.0, 1.0, steps):
                self.track(p0 + (np.asarray(position_world) - p0) * s, q0 if s == 0.0 else quat_wxyz)
        finally:
            self.max_step = step
        return self._joints() if self.last_error_mm < 5.0 else None

    def track(self, position_world, quat_wxyz):
        """One frame: the joints for the new target, starting from the previous solution. Returns {joint: angle}; when the target cannot be
        reached (outside the workspace, or in the table) the arm holds its last joints. After 5 failed frames in a row a multi-seed solve
        looks for another arm configuration, accepted only if it is close to the current one. last_error_mm is the position error."""
        goal = self._goal(position_world, quat_wxyz)
        result = self.tracker.solve_pose(goal, current_state=self.tracker.get_active_js(self.state).squeeze(1), return_seeds=1)
        if bool(result.success.any()):
            self.failures = 0
            self._move_to(result.js_solution)
            self.last_error_mm = float(result.position_error.reshape(-1)[0]) * 1000.0
        else:
            self.failures += 1
            if self.failures >= 5:
                full = self.full.solve_pose(goal)
                if bool(full.success.item()):
                    self._move_to(full.js_solution)
                    self.failures = 0
        return self._joints()

    def _move_to(self, solution):
        """Adopt `solution`, but move each joint by at most max_step."""
        old, new = self.state.position, solution.position
        step = (new - old).clamp(-self.max_step, self.max_step)
        self.state = solution.clone()
        self.state.position = old + step

    def _joints(self):
        # the solution includes the locked waist joints, so name the values from the solution itself
        return dict(zip(self.state.joint_names, self.state.position.reshape(-1).tolist()))

    def forward(self, joints):
        """Hand base pose in the world frame for {joint: angle}: (position, quaternion wxyz). For checks."""
        from curobo.types import JointState
        q = self.torch.tensor([[joints[n] for n in self.joint_names]], device='cuda', dtype=self.torch.float32)
        js = JointState.from_position(q, joint_names=self.joint_names)
        pose = self.tracker.compute_kinematics(js).tool_poses.to_dict()[self.frame]
        return pose.position.reshape(-1).cpu().numpy() + np.array([0.0, 0.0, PELVIS_Z]), pose.quaternion.reshape(-1).cpu().numpy()
