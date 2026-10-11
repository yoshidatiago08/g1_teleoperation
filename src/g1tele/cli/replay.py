"""g1tele-replay: re-retarget a recorded episode, offline.

Every episode keeps the operator's raw state (operator.msgpack). Replaying it through a different
retargeter setting, backend or robot gives a new episode without the operator doing anything again:

    g1tele-replay recordings/pick/episode_0003 --set ik.elbow_weight=0.5 --out recordings/pick_v2
"""
from __future__ import annotations

import argparse
import logging
import os
import time

from g1tele.config import load_config, parse_overrides
from g1tele.protocol import OperatorFrame
from g1tele.record import EpisodeRecorder, read_operator_stream
from g1tele.retarget.retargeter import Retargeter
from g1tele.robot.runner import make_backend


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episode', help='episode folder containing operator.msgpack')
    ap.add_argument('--config', '-c')
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--out', help='root folder of the new episode (default: next to the original)')
    ap.add_argument('--realtime', action='store_true', help='replay at the recorded pace (to watch it)')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config, parse_overrides(args.set))
    cfg.robot.rate_hz = min(cfg.robot.rate_hz, 250.0)
    frames = [OperatorFrame.from_bytes(b) for b in read_operator_stream(os.path.join(args.episode, 'operator.msgpack'))]
    if not frames:
        raise SystemExit('empty episode')
    retargeter, backend = Retargeter(cfg), make_backend(cfg)
    root = args.out or os.path.dirname(os.path.dirname(os.path.abspath(args.episode)))
    task = os.path.basename(os.path.dirname(os.path.abspath(args.episode))) + '_replay'
    rec = EpisodeRecorder(root, task, fps=cfg.record.fps, raw=True)
    rec.start()
    t0 = time.time()
    steps_per_frame = max(1, int(round(backend.rate_hz / cfg.record.fps)))
    for i, frame in enumerate(frames):
        targets = retargeter.step(frame, now=frame.t_capture)
        backend.apply(targets)
        for _ in range(steps_per_frame):
            backend.step()
        rec.add(frame.to_bytes(), targets, backend.observe(), None)
        if args.realtime:
            delay = t0 + (frame.t_capture - frames[0].t_capture) - time.time()
            if delay > 0:
                time.sleep(delay)
    print('wrote', rec.stop())
    backend.close()


if __name__ == '__main__':
    main()
