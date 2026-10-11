import pytest

from g1tele.config import load_config, parse_overrides


def test_defaults_load():
    cfg = load_config()
    assert cfg.robot.backend == 'mujoco'
    assert cfg.perception.arms == ['right', 'left']
    assert cfg.perception.waist is False       # off by default for manipulation datasets


def test_overrides_and_types():
    cfg = load_config(overrides=parse_overrides(['robot.backend=dds', 'perception.hands=false',
                                                 'perception.filters.wrist_beta=7', 'link.host=10.0.0.2']))
    assert cfg.robot.backend == 'dds' and cfg.perception.hands is False
    assert cfg.perception.filters.wrist_beta == 7.0 and cfg.link.host == '10.0.0.2'


def test_unknown_key_is_an_error():
    with pytest.raises(ValueError, match='unknown key'):
        load_config(overrides={'robot': {'backnd': 'dds'}})


def test_wrong_type_is_an_error():
    with pytest.raises(ValueError):
        load_config(overrides={'perception': {'hands': 'yes please'}})
    with pytest.raises(ValueError):
        load_config(overrides={'robot': {'rate_hz': 'fast'}})


def test_default_yaml_matches_defaults():
    """configs/default.yaml documents every default: it must stay loadable and equal to them."""
    import os
    from g1tele.config import REPO_ROOT
    assert load_config(os.path.join(REPO_ROOT, 'configs', 'default.yaml')).to_dict() == load_config().to_dict()
