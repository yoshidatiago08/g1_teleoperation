# dexhand: precise hand manipulation with dex-retargeting

Your hand in front of a camera -> the Inspire hand in RViz. Fingers are retargeted with
[dex-retargeting](https://github.com/dexsuite/dex-retargeting); the hand's position is **scaled and clutched** so a large motion
of yours is a small, precise motion of the robot hand. Step one of a fine-manipulation branch: arm IK comes later, so for now the
hand floats at its target.

This directory is self-contained (own Dockerfile, own compose); it shares nothing with the whole-body `g1tele` images.

## Run

```bash
cd dexhand
cp env.example .env                       # optional
xhost +local:docker
docker compose up --build                 # webcam; RViz and the camera window open
SOURCE=synthetic docker compose up        # no camera: a made-up hand drives everything
docker compose --profile test run --rm tests
```

Stand 1-2 m from the camera, hand visible. `HAND=left` for the left hand, `SOURCE=realsense` for a RealSense (depth).

## What you see

| In RViz | Meaning |
|---|---|
| The Inspire hand | **Where the robot would be**: your motion x `motion_scale`, your fingers retargeted |
| Cyan skeleton + `operator_hand` axes | Your tracked hand at **scale 1**, from the same starting point, to compare |

Camera window: `space` = clutch (the robot hand stops; move your hand somewhere comfortable; press again, it continues from where
it was, no jump). `+` / `-` = motion scale x1.25 / x0.8. Losing the hand acts as a clutch.
Live: `ros2 param set /dex_hand motion_scale 0.15`, `ros2 service call /dex_hand/toggle_clutch std_srvs/srv/Trigger`.

## How it works

```
camera -> MediaPipe Hands -> 21 landmarks
   fingers: wrist->fingertip vectors in the hand frame -> dex-retargeting (vector retargeting, URDF + mimic joints) -> 6 Inspire joints
   wrist:   pixel + depth (RealSense, or the hand's apparent size) -> operator axes -> MotionScaler (scale, clutch) -> hand target
```

- `frames.py` hand frame (x fingers, z palm normal), depth from hand size, camera/operator axes. `motion.py` the scaler/clutch.
  `retarget.py` the dex-retargeting wrapper. `pipeline.py` glues them; `node.py` is only ROS I/O. All but `node.py` run without ROS.
- Two different scales: `finger_scale` (dex's `scaling_factor`, your finger length vs the robot's, ~1.15) and `motion_scale`
  (hand travel, ~0.3). Rotation has its own `rotation_scale` (1.0 = turns like your hand).
- Difference from dex-retargeting's examples: they rotate landmarks into a MANO convention the URDF must share; here the human
  vectors are expressed in the robot hand's own base link (`retarget.robot_hand_rotation`), so the G1's hand URDF is used as is.
- `tools/make_hand_urdf.py` extracts the hands (with fingertip frames) from the G1 URDF into `ws/src/dex_hand/urdf/`.

## Status

Verified: unit tests (frames, scaler/clutch, dex-retargeting on both hand URDFs, full pipeline on a synthetic hand).
Not verified without hardware: MediaPipe on a real hand under numpy 2, the RealSense path, how the depth-from-hand-size estimate
behaves with a real webcam (set `HFOV_DEG` to your camera's field of view).
