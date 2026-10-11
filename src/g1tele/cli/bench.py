"""g1tele-bench: measure the perception pipeline.

    g1tele-bench synthetic                       # scripted operator with exact ground truth
    g1tele-bench synthetic --set perception.filters.wrist_beta=30 --set perception.bone_blend=0.5
    g1tele-bench synthetic --compare perception.filters.wrist_beta=8,30,60
    g1tele-bench closedloop --seconds 45        # scripted operator -> link -> retargeter -> MuJoCo
    g1tele-bench replay recordings/session1      # a recording made with `g1tele-perception --record-rgbd`
"""
from __future__ import annotations

import argparse
import json

from g1tele import bench
from g1tele.config import load_config, parse_overrides


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['synthetic', 'replay', 'closedloop'])
    ap.add_argument('path', nargs='?', help='recording folder (replay)')
    ap.add_argument('--config', '-c')
    ap.add_argument('--set', action='append', default=[], help='section.key=value override')
    ap.add_argument('--compare', help='key=v1,v2,v3: run once per value and print a table')
    ap.add_argument('--seconds', type=float, default=45.0)
    ap.add_argument('--tilt', type=float, default=0.0, help='synthetic camera tilt (deg)')
    ap.add_argument('--distance', type=float, default=1.8, help='synthetic operator distance (m)')
    ap.add_argument('--pixel-noise', type=float, default=1.2, help='synthetic 2D landmark noise (px); MediaPipe is closer to 3-5')
    ap.add_argument('--depth-noise', type=float, default=0.002, help='synthetic depth noise factor (sigma = f * z^2)')
    ap.add_argument('--radius-scale', type=float, default=1.0,
                    help='synthetic limbs thicker/thinner than the pipeline assumes')
    ap.add_argument('--json', help='write the metrics here')
    args = ap.parse_args(argv)

    base = parse_overrides(args.set)
    variants = [('run', {})]
    if args.compare:
        key, values = args.compare.split('=', 1)
        variants = [(f'{key.split(".")[-1]}={v}', parse_overrides([f'{key}={v}'])) for v in values.split(',')]
    rows = []
    for name, extra in variants:
        merged = dict(base)
        for k, v in extra.items():
            merged.setdefault(k, {})
            merged[k] = {**merged[k], **v} if isinstance(v, dict) else v
        cfg = load_config(args.config, _deep(base, extra))
        if args.mode == 'closedloop':
            rows.append((name, bench.run_closed_loop(cfg, args.seconds, args.tilt or 10.0)))
        elif args.mode == 'synthetic':
            rig, results = bench.run_synthetic(
                cfg, args.seconds, rig_kwargs=dict(tilt_deg=args.tilt, distance=args.distance,
                                                   radius_scale=args.radius_scale,
                                                   pixel_noise=args.pixel_noise, depth_noise=args.depth_noise))
            rows.append((name, bench.metrics_synthetic(rig, results)))
        else:
            rows.append((name, _replay(cfg, args.path)))
    print(bench.format_table(rows))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(dict(rows), f, indent=2)


def _deep(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _replay(cfg, path):
    from g1tele.perception.camera import ReplayCamera
    from g1tele.cli.perception import make_estimators
    from g1tele.perception.pipeline import PerceptionPipeline
    cam = ReplayCamera(path, realtime=False)
    pose, hands = make_estimators(cfg)
    pipe = PerceptionPipeline(cfg, pose, hands)
    results = []
    while True:
        f = cam.read()
        if f is None:
            break
        results.append((pipe.process(f), None))
    return bench.metrics_recorded(results)


if __name__ == '__main__':
    main()
