"""g1tele-robot: run next to the simulator (or the robot).

    g1tele-robot --set robot.backend=mujoco --set robot.render=true
    g1tele-robot --set robot.backend=dds --set robot.dds_interface=lo      # Isaac Sim (unitree_sim_isaaclab)
"""
from __future__ import annotations

import argparse
import logging

from g1tele.config import load_config, parse_overrides
from g1tele.robot.runner import RobotRunner


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', '-c')
    ap.add_argument('--set', action='append', default=[], help='section.key=value override')
    ap.add_argument('--confirm-real', action='store_true',
                    help="required with robot.target=real: this will move a physical robot")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    cfg = load_config(args.config, parse_overrides(args.set))
    if cfg.robot.target == 'real' and not args.confirm_real:
        raise SystemExit('robot.target=real moves a physical robot. Re-run with --confirm-real once the '
                         'area is clear and the emergency stop is in reach.')
    RobotRunner(cfg).run()


if __name__ == '__main__':
    main()
