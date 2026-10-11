# Architecture

```
perception (operator machine)                         robot host
camera ─► pose 2D ─► lift 3D ─► fusion ─► filters ─► OperatorFrame ══link══► Retargeter ─► backend
              │         │          │                       ▲                      │
         MediaPipe   depth +    bone lengths           protocol.py            urdf.py, arm_ik.py
                     radius     + rays + depth                                hand mapping
```

## Layers and what each may depend on

| Layer | Modules | Knows about |
|---|---|---|
| protocol / link | `protocol.py`, `link.py` | nothing else |
| perception | `perception/*` | cameras, MediaPipe, the operator. **Nothing about robots.** |
| retarget | `retarget/*` | the robot URDF. **Nothing about cameras.** |
| robot backends | `robot/*` | how to move one particular robot/simulator |

The seam is `OperatorFrame` (`protocol.py`): per arm the wrist and elbow **relative to the shoulder in
a gravity-aligned body frame** (metres, human scale), the hand frame, six finger curls
(0 open ... 1 closed, Inspire actuator order), status and quality; per operator the calibrated bone
lengths and optionally waist angles. About 1 kB, msgpack. Replace the perception side with anything
(a VR headset, gloves) that fills an `OperatorFrame` and everything downstream works unchanged.

## Perception, step by step (`perception/pipeline.py`)

1. **Pose** (`pose_mediapipe.py`): 2D landmarks with visibility. MediaPipe's own smoothing is off (it adds lag).
2. **Lift** (`lifting.py`): depth at the landmark, taken from the *near* part of a 9x9 window (35th percentile), moved along the viewing ray by the limb radius (the camera sees skin, not the joint centre). A landmark outside the image or with low visibility is unusable whatever MediaPipe's score.
3. **Shoulders**: both always measured; a reading faster than 1.2 m/s or breaking the shoulder width is rejected; a hidden shoulder is predicted from the other; a hidden pair is held for 3 s (shoulders barely move).
4. **Fusion** (`fusion.py`): per arm solve for the elbow's and wrist's distance along their pixel rays, from the depth readings (Tukey-robust), the calibrated bone lengths (σ 6 mm) and a weak temporal prior. Candidate starting points: the measured depths and both intersections of each ray with its bone sphere; the winner is chosen with a *wide* Cauchy score on the depth readings (σ 5 cm) so a reading a few cm off still points to the right root. A wrist solution faster than 5 m/s is re-solved anchored at the previous depths.
5. **Body frame**: columns forward, left, up. Left = shoulder line at calibration, up = IMU, floor plane, or calibration arms. Fixed after calibration: how you turn later does not tilt the arm targets.
6. **Filters** (`filters.py`): One Euro on wrist, elbow (relative to the shoulder) and on hand rotation and each finger.
7. **Hands** (`hand.py`, `tracker.HandTrack`): MediaPipe Hands on a crop around the wrist, run in parallel threads per hand. Frame x = fingers, z = palm normal; curls from joint angles scaled to the calibrated open/fist range.
8. **Status**: `TRACKING` (measured now), `HELD` (not measured for under 0.25 s, last value repeated with decaying quality), `LOST` (the robot eases to rest). Waist from the shoulders is optional (`perception.waist`).

## Retargeting (`retarget/`)

`Retargeter.step(frame)` returns `RobotTargets` (joint angles by URDF name, five-finger curls, status,
IK error).

- Robot kinematics come from the URDF (`urdf.py`, `arm_ik.py`): 7-joint chain per arm. The robot's *wrist point* is the origin of the wrist-pitch joint (it lies on the wrist-roll axis, so shoulder and elbow alone place it and the wrist joints only orient the hand). The "shoulder" is the centre of the best-fit sphere the elbow moves on (the G1's shoulder axes do not meet in one point, which would otherwise cost about 1 cm).
- `bone_scaled` mapping: the robot's elbow = shoulder + upper-arm length × direction of *your* upper arm; wrist = elbow + forearm length × direction of your forearm. Fallback without an elbow: uniform scaling by reach.
- IK: damped least squares, analytic Jacobians, joint limits by active set, warm start, about 2 ms per arm. Tasks: wrist position, elbow position (picks the posture), hand orientation, closeness to the previous solution (kept tiny: it biases tracking), faint pull to a relaxed posture.
- Safety: joint speed clamp (10 rad/s), `LOST` eases to rest, and the robot runner eases to rest when no frame arrives for 0.5 s.

## Robot host (`robot/runner.py`)

Reacts to a frame the moment it arrives (poll with the time to the next backend tick as timeout), ticks the
backend at its own rate (250 Hz), sends status (and JPEG of the simulator) back, records episodes.
A frame is never queued: the link keeps only the newest.

Clock note: `t_capture` is the operator machine's wall clock. Latency numbers across two machines
are only as good as their clock sync (use chrony/NTP); the closed-loop benchmark runs on one machine.
