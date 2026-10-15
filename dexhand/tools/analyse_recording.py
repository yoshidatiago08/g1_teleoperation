"""Study a recording made with `--record` (RECORD=... in .env): the hand model against the raw MediaPipe landmarks.

    python tools/analyse_recording.py recordings/take1.npz [--hand right] [--hand-length-mm 190] [--calibration 40:90] [--plot take1.png]

In the VM image:  docker compose --profile vm run --rm --entrypoint python vm tools/analyse_recording.py recordings/take1.npz
(set PYTHONPATH=src if you run it outside the image)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from dex_hand.analysis import analyse, plot, report
from dex_hand.record import load

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('recording')
parser.add_argument('--hand', default='right', choices=['right', 'left'])
parser.add_argument('--hand-length-mm', type=float, default=None)
parser.add_argument('--calibration', default=None, help='FIRST:LAST frame indices of the stretched palm-to-camera hand (default: the frames flagged while calibrating)')
parser.add_argument('--plot', default=None, help='save a plot (PNG)')
args = parser.parse_args()
calibration = None if args.calibration is None else tuple(int(v) for v in args.calibration.split(':'))
result = analyse(load(args.recording), args.hand, None if args.hand_length_mm is None else args.hand_length_mm / 1000.0, calibration)
print(report(result))
if args.plot:
    plot(result, args.plot)
    print('plot:', args.plot)
