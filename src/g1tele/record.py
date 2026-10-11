"""Episode recording.

Each episode is a folder:
    data.json        same schema as unitree's xr_teleoperate / unitree_sim_isaaclab episodes
                     (info, text, data[] with colors, states, actions, sim_state), so their replay
                     and data-generation tools work on it
    operator.msgpack every OperatorFrame received (raw, before retargeting): with it an episode can
                     be re-retargeted offline with a better retargeter or other robot
    quality.json     per-step tracking status, IK error and latency, to filter bad segments
    colors/          simulator camera frames, when the backend provides them
"""
from __future__ import annotations

import datetime
import json
import os
import time
from typing import Dict, Optional

import cv2
import msgpack
import numpy as np

from g1tele.retarget.retargeter import RobotTargets
from g1tele.retarget.urdf import arm_joint_names, hand_joint_names
from g1tele.robot.base import curl_to_dds


class EpisodeRecorder:
    def __init__(self, root: str, task: str = 'task', goal: str = '', fps: float = 30.0,
                 sides=('left', 'right'), raw: bool = True):
        self.root, self.task, self.goal, self.fps = root, task, goal, fps
        self.sides = list(sides)
        self.raw = raw
        self.dir: Optional[str] = None
        self.active = False

    def _next_dir(self) -> str:
        task_dir = os.path.join(self.root, self.task)
        os.makedirs(task_dir, exist_ok=True)
        done = [d for d in os.listdir(task_dir) if d.startswith('episode_')]
        idx = 1 + max([int(d.split('_')[-1]) for d in done], default=-1)
        return os.path.join(task_dir, f'episode_{idx:04d}')

    def start(self):
        self.dir = self._next_dir()
        os.makedirs(os.path.join(self.dir, 'colors'), exist_ok=True)
        self.items, self.quality = [], []
        self.raw_file = open(os.path.join(self.dir, 'operator.msgpack'), 'wb') if self.raw else None
        self.t0 = time.time()
        self.active = True

    def _arm(self, q: Dict[str, float], side: str):
        return [float(q.get(n, 0.0)) for n in arm_joint_names(side)]

    def add(self, frame_bytes: Optional[bytes], targets: RobotTargets, observed: Dict[str, float],
            image: Optional[np.ndarray] = None, latency_ms: float = 0.0):
        if not self.active:
            return
        idx = len(self.items)
        colors = {}
        if image is not None:
            name = f'{idx:06d}_color_0.jpg'
            cv2.imwrite(os.path.join(self.dir, 'colors', name), image[:, :, ::-1])
            colors['color_0'] = os.path.join('colors', name)

        def block(arm_values, side, hand):
            return {'qpos': arm_values, 'qvel': [], 'torque': []}

        states, actions = {}, {}
        for side in self.sides:
            hand_obs = [float(observed.get(n, 0.0)) for n in hand_joint_names(side)]
            hand_cmd = curl_to_dds(targets.hand.get(side, np.zeros(6))).tolist()
            states[f'{side}_arm'] = block(self._arm(observed, side), side, None)
            actions[f'{side}_arm'] = block(self._arm(targets.arm_q, side), side, None)
            states[f'{side}_ee'] = block(hand_obs, side, None)
            actions[f'{side}_ee'] = block(hand_cmd, side, None)
        states['body'] = {'qpos': [float(observed.get(n, 0.0)) for n in targets.waist_q]}
        actions['body'] = {'qpos': [float(v) for v in targets.waist_q.values()]}
        self.items.append({'idx': idx, 'colors': colors or None, 'depths': None, 'states': states,
                           'actions': actions, 'tactiles': None, 'audios': None, 'sim_state': None})
        self.quality.append({'t': time.time() - self.t0, 'status': targets.status,
                             'ik_error_mm': {s: round(1000 * e, 3) for s, e in targets.ik_error.items()},
                             'latency_ms': round(latency_ms, 2), 'seq': targets.seq})
        if self.raw_file is not None and frame_bytes is not None:
            self.raw_file.write(msgpack.packb(frame_bytes, use_bin_type=True))

    def stop(self, save: bool = True) -> Optional[str]:
        if not self.active:
            return None
        self.active = False
        if self.raw_file is not None:
            self.raw_file.close()
        if not save or not self.items:
            import shutil
            shutil.rmtree(self.dir, ignore_errors=True)
            return None
        arm_names = {s: arm_joint_names(s) for s in self.sides}
        info = {'version': '1.0.0', 'date': datetime.date.today().strftime('%Y-%m-%d'),
                'author': 'g1tele', 'image': {'width': 0, 'height': 0, 'fps': self.fps},
                'depth': {'width': 0, 'height': 0, 'fps': self.fps},
                'joint_names': {**{f'{s}_arm': arm_names[s] for s in self.sides},
                                **{f'{s}_ee': hand_joint_names(s) for s in self.sides}, 'body': []},
                'tactile_names': {'left_ee': [], 'right_ee': []}, 'sim_state': ''}
        with open(os.path.join(self.dir, 'data.json'), 'w', encoding='utf-8') as f:
            json.dump({'info': info,
                       'text': {'goal': self.goal, 'desc': self.task, 'steps': ''},
                       'data': self.items}, f)
        with open(os.path.join(self.dir, 'quality.json'), 'w', encoding='utf-8') as f:
            json.dump(self.quality, f)
        return self.dir


def read_operator_stream(path: str):
    """Yield the raw OperatorFrame bytes stored in an operator.msgpack file."""
    with open(path, 'rb') as f:
        for item in msgpack.Unpacker(f, raw=False):
            yield item
