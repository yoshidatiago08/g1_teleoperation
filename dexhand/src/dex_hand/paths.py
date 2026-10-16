"""Where the URDFs, dex-retargeting configs and RViz config live (dexhand/assets, also /app/assets in the images).

Two robot hands can be mounted on the G1 (`--hand-model`, env HAND_MODEL): the Inspire hand (6 actuators) and the Unitree Dex5-1 (16 actuators).
"""
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[2] / 'assets'
URDF_DIR = ASSETS / 'urdf'
HAND_MODELS = ('inspire', 'dex5')
DEFAULT_HAND_MODEL = 'inspire'
_FILE = {'inspire': 'inspire_hand', 'dex5': 'dex5_hand'}


def _check(model):
    if model not in HAND_MODELS:
        raise ValueError(f'unknown hand model {model!r}: choose from {HAND_MODELS}')
    return model


def urdf(side, model=DEFAULT_HAND_MODEL):
    return URDF_DIR / f'{_FILE[_check(model)]}_{side}.urdf'


def retarget_config(side, model=DEFAULT_HAND_MODEL):
    return ASSETS / 'config' / f'{_FILE[_check(model)]}_{side}.yml'


def g1_urdf(model=DEFAULT_HAND_MODEL):
    """The G1 without legs, with this hand mounted."""
    return ASSETS / 'g1' / f'g1_upper_{_check(model)}.urdf'


def curobo_config(model=DEFAULT_HAND_MODEL):
    return ASSETS / 'curobo' / ('g1_right_arm.yml' if _check(model) == 'inspire' else f'g1_right_arm_{model}.yml')


G1_URDF = g1_urdf('inspire')
G1_MESHES = ASSETS / 'g1' / 'meshes'
CUROBO_CONFIG = curobo_config('inspire')
