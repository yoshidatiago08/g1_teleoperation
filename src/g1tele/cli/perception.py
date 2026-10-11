"""g1tele-perception: run on the machine with the camera.

    g1tele-perception --set link.host=100.64.0.7            # live, window, sends to the robot host
    g1tele-perception --source synthetic --no-gui           # no camera: scripted operator
    g1tele-perception --record-rgbd recordings/session1     # also store the camera stream (for g1tele-bench replay)

Keys in the window: c = full calibration, v = quick calibration (rest pose only), o = reset the cylinder, x = record the camera stream for debugging, r = start/stop an
episode recording on the robot host, q = quit.
"""
from __future__ import annotations

import argparse
import logging
import os
import threading
import time

import cv2
import numpy as np

from g1tele import protocol
from g1tele.config import load_config, parse_overrides
from g1tele.link import OperatorLink
from g1tele.perception import overlay
from g1tele.perception.calibration import Profile
from g1tele.perception.camera import LatestFrame, RGBDRecorder, ReplayCamera
from g1tele.perception.pipeline import PerceptionPipeline

log = logging.getLogger('g1tele.perception')


def make_estimators(cfg):
    try:
        from g1tele.perception.pose_mediapipe import MediaPipeHands, MediaPipePose
    except ImportError as e:      # pragma: no cover
        raise SystemExit(f'mediapipe is not installed ({e}). pip install "g1tele[perception]"')
    pose = MediaPipePose(cfg.perception.pose_model_complexity, smooth=cfg.perception.pose_smoothing)
    hands = ({side: MediaPipeHands(cfg.perception.hand_crop_margin) for side in cfg.perception.arms}
             if cfg.perception.hands else None)
    return pose, hands


def make_source(cfg, name, replay_path):
    if name == 'realsense':
        from g1tele.perception.camera import RealSenseCamera
        return RealSenseCamera(cfg.camera)
    if name == 'replay':
        return ReplayCamera(replay_path or cfg.camera.replay_path, realtime=True)
    if name == 'synthetic':
        from g1tele.perception.synthetic import SyntheticOperator, SyntheticRig
        op = SyntheticOperator(poses=tuple(cfg.calibration.poses), hand_poses=cfg.calibration.hand_poses)
        return SyntheticRig(op, realtime=True, tilt_deg=10.0)
    raise SystemExit(f'unknown source {name!r}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', '-c')
    ap.add_argument('--set', action='append', default=[], help='section.key=value override')
    ap.add_argument('--source', choices=['realsense', 'replay', 'synthetic'])
    ap.add_argument('--replay', help='folder written by --record-rgbd')
    ap.add_argument('--no-gui', action='store_true')
    ap.add_argument('--record-rgbd', help='store the camera stream in this folder')
    ap.add_argument('--operator', help='profile name (calibration is stored per operator)')
    ap.add_argument('--full-calibration', action='store_true', help='ignore a saved profile and redo all poses')
    ap.add_argument('--task', default='task', help='task name for episode recordings')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')

    overrides = parse_overrides(args.set)
    if args.source:
        overrides.setdefault('camera', {})['source'] = args.source
    if args.operator:
        overrides.setdefault('calibration', {})['operator'] = args.operator
    cfg = load_config(args.config, overrides)

    source = make_source(cfg, cfg.camera.source, args.replay)
    if cfg.camera.source == 'synthetic':
        pose, hands = source.pose, (source.hands if cfg.perception.hands else None)
    else:
        pose, hands = make_estimators(cfg)
    profile = None if args.full_calibration else Profile.load(cfg.calibration.profile_dir, cfg.calibration.operator)
    pipe = PerceptionPipeline(cfg, pose, hands, profile)
    if profile is None:
        log.info('no saved profile for %r: full calibration', cfg.calibration.operator)
    link = OperatorLink(cfg.link)
    grabber = LatestFrame(source)
    rec = {'r': RGBDRecorder(args.record_rgbd, cfg.to_dict()) if args.record_rgbd else None}

    def toggle_rgbd():
        """x: start/stop recording the camera stream (for debugging with replay)."""
        if rec['r'] is None:
            path = os.path.join(cfg.record.root, 'rgbd', time.strftime('%Y%m%d_%H%M%S'))
            rec['r'] = RGBDRecorder(path, cfg.to_dict())
            log.info('recording the camera stream to %s', path)
        else:
            r, rec['r'] = rec['r'], None
            r.close(None if pipe.profile is None else pipe.profile.to_full())
            log.info('camera recording saved: %s (%d frames, %.0f MB)', r.path, r.n, r.bytes / 1e6)
    gui = not args.no_gui
    stop = threading.Event()
    latest = {'seq': -1}
    lock = threading.Lock()

    def worker():
        """Camera -> perception -> link, as fast as the camera and the CPU allow. Nothing here waits for
        the window or for the robot's video."""
        n, t_log = 0, time.time()
        while not stop.is_set():
            frame = grabber.get(timeout=0.5)
            if frame is None:
                if grabber._done:
                    stop.set()
                continue
            out = pipe.process(frame)
            link.send_frame(out)
            r = rec['r']
            if r is not None:
                if pipe.profile is not None and pipe.profile.frame is not None:
                    r.set_start_profile(pipe.profile.to_full())      # the calibration this recording starts with
                r.write(frame, out.to_bytes())
            with lock:
                latest.update(seq=out.seq, color=frame.color, out=out, debug=dict(pipe.last_debug))
            n += 1
            if time.time() - t_log > 5.0:
                log.info('%.1f fps  %s  %s', n / (time.time() - t_log), out.mode,
                         {s: a.status for s, a in out.arms.items()})
                n, t_log = 0, time.time()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    feedback, recording, shown_seq, robot_frames, t_fps = None, False, -1, 0, time.time()
    try:
        while not stop.is_set():
            fb = link.poll_feedback()
            if fb is not None:
                feedback = fb
            if not gui:
                time.sleep(0.01)
                continue
            if fb is not None and fb.get('jpeg'):
                sim = cv2.imdecode(np.frombuffer(fb['jpeg'], np.uint8), cv2.IMREAD_COLOR)
                if sim is not None:
                    cv2.imshow('robot', sim)
                    robot_frames += 1
            with lock:
                seq = latest['seq']
                snap = None if seq == shown_seq else dict(latest)
            if snap is not None:
                shown_seq = seq
                cv2.imshow('g1tele perception',
                           overlay.draw(snap['color'], snap['out'], snap['debug'], feedback,
                                        '' if rec['r'] is None else f"REC camera {rec['r'].n} fr {rec['r'].bytes / 1e6:.0f} MB"))
            if time.time() - t_fps > 5.0:
                log.info('robot video %.0f fps', robot_frames / (time.time() - t_fps))
                robot_frames, t_fps = 0, time.time()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                pipe.start_calibration(full=True)
            elif key == ord('v'):
                pipe.start_calibration(full=False)
            elif key == ord('o'):
                link.send_event(cmd='reset')          # put the cylinder back
            elif key == ord('x'):
                toggle_rgbd()
            elif key == ord('r'):
                recording = not recording
                link.send_event(cmd='record_start' if recording else 'record_stop', task=args.task)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        thread.join(timeout=2.0)
        if recording:
            link.send_event(cmd='record_stop')
        grabber.close()
        if rec['r'] is not None:
            rec['r'].close(None if pipe.profile is None else pipe.profile.to_full())
            log.info('camera recording saved: %s (%d frames)', rec['r'].path, rec['r'].n)
        link.close()
        pipe.close()
        if gui:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
