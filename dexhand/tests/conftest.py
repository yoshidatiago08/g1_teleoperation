import sys
from pathlib import Path

# The package is dexhand/src/dex_hand; in the Docker images it is /app/src/dex_hand.
SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
