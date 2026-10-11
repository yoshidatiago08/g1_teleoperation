# Adding a simulator or a robot

A backend is one class (`robot/base.py`):

```python
class MySim(RobotBackend):
    rate_hz = 100.0                         # how often step() is called

    def __init__(self, cfg: Config): ...    # connect / load the scene

    def apply(self, targets: RobotTargets): # newest targets: cheap, just store them
        # targets.arm_q    {'right_shoulder_pitch_joint': rad, ...}   14 arm joints, URDF names
        # targets.waist_q  {'waist_yaw_joint': rad, ...}
        # targets.hand     {'right': 6 curls, 'left': 6 curls}        0 open ... 1 closed
        #                  robot.base.curl_to_joint(...) -> Inspire joint angles (rad)
        #                  robot.base.curl_to_dds(...)   -> the [0, 1] DDS command (1 = open)
        ...

    def step(self): ...                     # advance one control period
    def observe(self) -> dict: ...          # optional: joint positions by URDF name (recording)
    def render(self): ...                   # optional: RGB image sent back to the operator
```

Register it in `robot/runner.py: make_backend` and add its name to `RobotCfg.backend`.

Other robots: the retargeter reads the arm chains from the URDF in `robot.urdf` (joint names
`{side}_shoulder_pitch_joint ... {side}_wrist_yaw_joint`, links `torso_link` and `{side}_wrist_yaw_link`)
and the hand joints from `retarget/urdf.py: HAND_ACTUATOR_JOINT`. A robot with other joint names needs
those two lists adapted; the perception side does not change.

Another input device (VR, gloves): produce `protocol.OperatorFrame`s and send them with `link.OperatorLink`.
