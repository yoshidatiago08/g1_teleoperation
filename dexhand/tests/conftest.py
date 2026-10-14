import sys
from pathlib import Path

# In the repo the package sits next to the tests' parent; in the Docker image it is built at /ws/src/dex_hand.
PKG = Path(__file__).resolve().parents[1] / 'ws' / 'src' / 'dex_hand'
if not PKG.exists():
    PKG = Path('/ws/src/dex_hand')
sys.path.insert(0, str(PKG))
