"""Benchmark the perception pipeline: accuracy, jitter, lag, dropouts and speed.

With a synthetic operator the truth is exact, so accuracy and lag are measured directly. With a
recorded session (no truth) it reports jitter, dropouts and timing, which is enough to compare
two filter settings on the same data.
"""
from __future__ import annotations

import tempfile
from typing import Dict, List, Optional

import numpy as np

from g1tele import protocol
from g1tele.config import Config, load_config
from g1tele.perception.pipeline import PerceptionPipeline
from g1tele.perception.synthetic import SyntheticOperator, SyntheticRig


def run_synthetic(cfg: Config, seconds: float = 45.0, rig_kwargs: Optional[dict] = None,
                  op_kwargs: Optional[dict] = None, occlusions=None):
    """Runs the pipeline on a scripted operator. Returns (rig, [(OperatorFrame, Truth)])."""
    cfg.calibration.profile_dir = tempfile.mkdtemp()     # a benchmark must not reuse a saved profile
    op = SyntheticOperator(poses=tuple(cfg.calibration.poses), hand_poses=cfg.calibration.hand_poses,
                           **(op_kwargs or {}))
    if occlusions:
        op.occlusions = list(occlusions)
    rig = SyntheticRig(op, **(rig_kwargs or {}))
    pipe = PerceptionPipeline(cfg, rig.pose, rig.hands if cfg.perception.hands else None)
    results = []
    while rig.index * rig.dt < seconds:
        frame = rig.read()
        results.append((pipe.process(frame), rig.truth))
    rig.pipe = pipe
    return rig, results


def _static_windows(op: SyntheticOperator, t_end: float):
    """Time ranges in which the scripted arms hold still (the end of each calibration pose)."""
    acc, windows = 0.0, []
    for name, dur in op.script:
        windows.append((acc + dur - 1.2, acc + dur - 0.1))
        acc += dur
    return [w for w in windows if w[1] < t_end]


def metrics_synthetic(rig: SyntheticRig, results) -> Dict[str, float]:
    op, dt = rig.op, rig.dt
    out: Dict[str, float] = {}
    first = next((i for i, (o, _) in enumerate(results) if o.mode == protocol.TRACKING), None)
    if first is None:
        return {'calibrated': 0.0}
    out['calibration_s'] = round(first * dt, 1)
    sides = ('right', 'left')

    def err(lag_frames, part):
        e = []
        for i in range(first + 30, len(results)):
            o, tr = results[i]
            if tr.t < op.free_start or i - lag_frames < 0:
                continue
            tt = results[i - lag_frames][1]
            for s in sides:
                a = o.arms.get(s)
                if a is None or a.status != protocol.TRACKING or getattr(a, part) is None:
                    continue
                k = 'W' if part == 'wrist' else 'E'
                e.append(np.linalg.norm(getattr(a, part) - (tt.joints[s][k] - tt.joints[s]['S'])))
        return np.array(e)

    for part in ('wrist', 'elbow'):
        e = err(0, part) * 1000
        out[f'{part}_rms_mm'] = round(float(np.sqrt(np.mean(e ** 2))), 1)
        out[f'{part}_p95_mm'] = round(float(np.percentile(e, 95)), 1)
    # lag: the delay of the truth that fits the output best
    best_lag, best = 0, 1e9
    for lag in range(0, 8):
        e = err(lag, 'wrist')
        if len(e) and np.sqrt(np.mean(e ** 2)) < best:
            best, best_lag = np.sqrt(np.mean(e ** 2)), lag
    out['lag_ms'] = round(best_lag * dt * 1000, 0)
    out['wrist_rms_after_lag_mm'] = round(best * 1000, 1)
    # jitter: error scatter while the scripted arms hold still (the truth does not move, so any spread is noise)
    jit = []
    for (a, b) in _static_windows(op, results[-1][1].t):
        vals = {s: [] for s in sides}
        for o, tr in results[first:]:
            if a <= tr.t <= b:
                for s in sides:
                    arm = o.arms.get(s)
                    if arm is not None and arm.status == protocol.TRACKING and arm.wrist is not None:
                        vals[s].append(arm.wrist - (tr.joints[s]['W'] - tr.joints[s]['S']))
        for s in sides:
            if len(vals[s]) > 8:
                jit.append(float(np.sqrt(np.var(np.array(vals[s]), axis=0).sum())))
    out['still_jitter_mm'] = round(1000 * float(np.mean(jit)), 2) if jit else float('nan')
    # fingers
    fe = []
    for o, tr in results[first + 30:]:
        for s in sides:
            a = o.arms.get(s)
            if a is not None and a.fingers is not None and tr.t >= op.free_start:
                fe.append(np.mean(np.abs(a.fingers - tr.curls[s])))
    if fe:
        out['finger_mae'] = round(float(np.mean(fe)), 3)
    # palm orientation (the angle between the measured and the true hand frame)
    re = []
    for o, tr in results[first + 30:]:
        for s in sides:
            a = o.arms.get(s)
            if a is not None and a.hand_R is not None and tr.t >= op.free_start:
                re.append(np.degrees(np.arccos(np.clip((np.trace(a.hand_R.T @ tr.hand_R[s]) - 1) / 2, -1, 1))))
    if re:
        out['hand_rot_mean_deg'] = round(float(np.mean(re)), 1)
        out['hand_rot_p95_deg'] = round(float(np.percentile(re, 95)), 1)
    out.update(_status_and_timing(results))
    return out


def _status_and_timing(results) -> Dict[str, float]:
    tracking = [o for o, _ in results if o.mode == protocol.TRACKING]
    n = max(1, sum(len(o.arms) for o in tracking))
    out: Dict[str, float] = {}
    for st in (protocol.TRACKING, protocol.HELD, protocol.LOST):
        out[f'{st.lower()}_pct'] = round(100 * sum(a.status == st for o in tracking for a in o.arms.values()) / n, 1)
    totals = np.array([o.timing_ms.get('total', 0.0) for o, _ in results])
    out['frame_ms_mean'] = round(float(totals.mean()), 2)
    out['frame_ms_p95'] = round(float(np.percentile(totals, 95)), 2)
    return out


def metrics_recorded(results) -> Dict[str, float]:
    """Metrics that need no truth."""
    tracking = [o for o, _ in results if o.mode == protocol.TRACKING]
    out = _status_and_timing(results)
    for s in ('right', 'left'):
        w = np.array([o.arms[s].wrist for o in tracking
                      if s in o.arms and o.arms[s].status == protocol.TRACKING and o.arms[s].wrist is not None])
        if len(w) > 10:
            d3 = np.diff(w, n=3, axis=0)       # third difference: removes smooth motion, keeps noise
            out[f'{s}_hf_noise_mm'] = round(float(1000 * np.sqrt(np.mean(np.sum(d3 ** 2, axis=1)) / 20)), 2)
    return out


def format_table(rows: List[tuple]) -> str:
    keys = sorted({k for _, m in rows for k in m})
    width = max(len(n) for n, _ in rows)
    lines = [' ' * width + '  ' + '  '.join(f'{k:>14}' for k in keys)]
    for name, m in rows:
        lines.append(name.ljust(width) + '  ' + '  '.join(f'{m.get(k, float("nan")):>14}' for k in keys))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------- closed loop
def run_closed_loop(cfg: Config, seconds: float = 45.0, tilt_deg: float = 10.0, sample_hz: float = 100.0):
    """Scripted operator -> perception -> ZeroMQ link -> retargeter -> robot backend, all live.

    Measures how well the robot's wrist follows the operator's TRUE wrist (mapped with the same
    bone-scaled rule) and the delay between them. Returns a metrics dict.
    """
    import random
    import threading
    import time

    from g1tele.geometry import unit
    from g1tele.link import OperatorLink
    from g1tele.retarget.arm_ik import ArmKinematics
    from g1tele.robot.runner import RobotRunner

    port = random.randint(20000, 40000)
    cfg.link.host, cfg.link.stream_port, cfg.link.control_port, cfg.link.feedback_port = \
        '127.0.0.1', port, port + 1, port + 2
    cfg.calibration.profile_dir = tempfile.mkdtemp()
    cfg.robot.render = False
    cfg.robot.scene = False         # the benchmark measures tracking: a table in the way would corrupt it
    runner = RobotRunner(cfg)
    stop = threading.Event()
    robot_thread = threading.Thread(target=runner.run, kwargs=dict(stop=stop), daemon=True)
    robot_thread.start()
    time.sleep(0.3)

    op = SyntheticOperator(poses=tuple(cfg.calibration.poses), hand_poses=cfg.calibration.hand_poses)
    rig = SyntheticRig(op, realtime=True, tilt_deg=tilt_deg, palm_plate=False)    # the arm benchmark: keep the frame inside its real-time budget
    pipe = PerceptionPipeline(cfg, rig.pose, rig.hands if cfg.perception.hands else None)
    link = OperatorLink(cfg.link)
    stop_sender = threading.Event()

    def sender():
        while not stop_sender.is_set():
            frame = rig.read()
            link.send_frame(pipe.process(frame))
    sender_thread = threading.Thread(target=sender, daemon=True)
    sender_thread.start()

    kin = {s: ArmKinematics(runner.retargeter.urdf, s) for s in cfg.perception.arms}
    samples = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        samples.append((time.time(), runner.backend.observe(), runner.latest_targets))
        time.sleep(1.0 / sample_hz)
    stop_sender.set()
    stop.set()
    sender_thread.join(timeout=2)
    robot_thread.join(timeout=3)
    link.close()

    def expected(side, script_t):
        tr = op.at(script_t)
        k = kin[side]
        e = tr.joints[side]['E'] - tr.joints[side]['S']
        w = tr.joints[side]['W'] - tr.joints[side]['E']
        return k.shoulder + k.upper_arm * unit(e) + k.forearm * unit(w)

    def robot_wrist(side, q):
        k = kin[side]
        return k.fk(np.array([q.get(n, 0.0) for n in k.names]))[0]

    free_all = [s for s in samples if s[0] - rig.t0_wall > op.free_start + 3 and s[1] and s[2] is not None]
    out: Dict[str, float] = {'samples': float(len(free_all))}
    if not free_all:
        out['calibrated'] = 0.0
        return out
    # errors are judged while the arm is being tracked: a hand outside the camera's view cannot be followed
    free = [s for s in free_all if s[2].mode == protocol.TRACKING]
    n_arm = sum(len(s[2].status) for s in free)
    out['tracking_pct'] = round(100.0 * sum(st == protocol.TRACKING for s in free for st in s[2].status.values())
                                / max(1, n_arm), 1)
    best = None
    for lag_ms in range(0, 400, 10):
        errs = []
        for t, q, tg in free:
            for side in kin:
                if tg.status.get(side) != protocol.TRACKING:
                    continue
                errs.append(np.linalg.norm(robot_wrist(side, q) - expected(side, t - rig.t0_wall - lag_ms / 1000.0)))
        rms = float(np.sqrt(np.mean(np.square(errs))))
        if best is None or rms < best[1]:
            best = (lag_ms, rms, float(np.percentile(errs, 95)))
    out['robot_wrist_vs_truth_rms_mm'] = round(best[1] * 1000, 1)
    out['robot_wrist_vs_truth_p95_mm'] = round(best[2] * 1000, 1)
    out['end_to_end_lag_ms'] = float(best[0])
    st = [s[2].ik_error for s in free if s[2] is not None]
    out['ik_error_mm_mean'] = round(1000 * float(np.mean([v for d in st for v in d.values()])), 2)
    out['robot_frames'] = float(runner.stats['frames'])
    out['apply_ms_last'] = round(runner.stats['apply_ms'], 2)
    return out
