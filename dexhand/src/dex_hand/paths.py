"""Where the URDFs, dex-retargeting configs and RViz config live (dexhand/assets, also /app/assets in the images)."""
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[2] / 'assets'
URDF_DIR = ASSETS / 'urdf'


def urdf(side):
    return URDF_DIR / f'inspire_hand_{side}.urdf'


def retarget_config(side):
    return ASSETS / 'config' / f'inspire_hand_{side}.yml'


G1_URDF = ASSETS / 'g1' / 'g1_upper_inspire.urdf'           # the G1 without legs, Inspire hands
G1_MESHES = ASSETS / 'g1' / 'meshes'
CUROBO_CONFIG = ASSETS / 'curobo' / 'g1_right_arm.yml'
