# dexhand: precise hand manipulation with dex-retargeting

Your hand in front of a camera -> a floating Inspire hand in MuJoCo (with a table and the red cylinder) and RViz. Fingers are
retargeted with [dex-retargeting](https://github.com/dexsuite/dex-retargeting); the hand's position is **scaled and clutched**, so a
large motion of yours is a small, precise motion of the robot hand. No arm yet: the hand floats, to judge fine control. Arm IK
(CuRobo) comes later, on the VM.

Same split as the main project: **the VM does everything heavy, the notebook only the camera and the display.**
This directory is self-contained (own Docker images, own compose); it shares nothing with the whole-body `g1tele` images.

```
 NOTEBOOK (light, no GPU)                               VM (heavy)
 ┌─────────────────────────────┐   JPEG frames        ┌──────────────────────────────────────────┐
 │ camera ─► JPEG ─────────────┼────────────────────► │ MediaPipe ─► dex-retargeting ─► scaler   │
 │                             │   (+ small depth)    │              (+ CuRobo later)    │       │
 │ window: preview + simulator │ ◄────────────────────┤ MuJoCo physics + offscreen render ◄──────┘
 │ RViz (hand, table, cylinder)│   state (1 kB) +     │                                          │
 │ keys: clutch, scale, reset ─┼─► events             └──────────────────────────────────────────┘
 └─────────────────────────────┘
```

## Run

```bash
cp env.example .env          # on both machines; on the notebook set VM_HOST to the VM's IP
```

**VM:**
```bash
cd dexhand && docker compose --profile vm up --build
```
**Notebook:**
```bash
cd dexhand && xhost +local:docker && docker compose --profile notebook up --build
```

No camera yet? VM: `VM_SOURCE=synthetic docker compose --profile vm up`, notebook: `CAMERA=none docker compose --profile notebook up`
(a made-up hand drives everything). `HAND=left`, `CAMERA=realsense` (sends depth too) go in `.env`.
The VM needs the NVIDIA container toolkit for the simulator video; without a GPU set `MUJOCO_GL=osmesa` (software rendering).
Ports 5601-5603 must be reachable from the notebook to the VM.

Stand 1-2 m from the camera, hand visible.

## What you see

A window with your camera (the VM draws the tracked hand on it) next to the simulator view, plus RViz. In MuJoCo the hand collides:
it pushes the cylinder and cannot go through the table. Text shows the state, the latency and `cylinder -> target X cm, lifted Y cm`.

| In RViz | Meaning |
|---|---|
| The Inspire hand | the **simulated** hand (what physics allowed) |
| `hand_target` axes | the scaled target the simulated hand is pulled towards |
| Cyan skeleton + `operator_hand` axes | your tracked hand at **scale 1** from the same starting point, to compare |

Keys in the window: `space` clutch (the robot hand stops, reposition your hand, press again: it continues from where it was, no
jump), `+` / `-` motion scale x1.25 / x0.8, `o` cylinder back, `q` quit. Losing the hand acts as a clutch.

## How it works

```
frames ─► MediaPipe ─► wrist -> fingertip vectors ─► dex-retargeting (URDF + mimic joints) ─► 6 Inspire joints ─► MuJoCo servos
                  └─► wrist pixel + depth ─► operator axes ─► MotionScaler (scale, clutch) ─► weld target ─► MuJoCo hand
```

- `src/dex_hand/` (pure Python, no ROS): `frames.py` hand frame and axes, `motion.py` scaler/clutch, `retarget.py` dex-retargeting
  wrapper, `pipeline.py` glue, `mujoco_scene.py` the scene, `vm.py` the VM app, `notebook.py` + `capture.py` the notebook app,
  `link.py` the ZeroMQ link, `ros_bridge.py` RViz (notebook only, optional).
- Two different scales: `FINGER_SCALE` (dex's `scaling_factor`, your finger length vs the robot's, ~1.15) and `MOTION_SCALE` (hand
  travel, ~0.3). Rotation has its own `ROTATION_SCALE` (1.0 = turns like your hand).
- Depth: RealSense depth at the wrist (sent at half size), or with a webcam estimated from the hand's apparent size (set `HFOV_DEG`).
- Differences from dex-retargeting's examples: they rotate landmarks into a MANO convention the URDF must share; here the human
  vectors are expressed in the robot hand's own base link (`retarget.robot_hand_rotation`), so the G1's hand URDF is used as is.
- MuJoCo: the hand is a free body held by a soft weld to the target; the weld's pull is capped at 3 cm and the contacts are stiff, so
  a hand blocked by the table presses on it instead of sinking in.
- Bandwidth: about 40 kB per colour frame, so roughly 1.5 MB/s at 30 fps, plus the state and two preview JPEGs coming back.
  The link drops stale frames instead of queueing them.
- `tools/make_hand_urdf.py` extracts the hands (with fingertip frames) from the G1 URDF into `assets/urdf/`.

Tests: `docker compose --profile test run --rm tests` (VM image; software rendering).

## Status

Verified (on the development machine, no camera, no screen): the unit and integration tests in both images (physics, link, frame
encoding, the whole VM app, the ROS bridge), the VM and notebook containers running against each other with the synthetic hand
(about 28 states/s, simulator video, clutch and scale events, `/joint_states`, TF and markers at about 27 Hz), and offscreen
rendering both on a GPU (EGL) and in software (osmesa).
Not verified: the RViz window and the notebook window (no display there), MediaPipe on a real hand under numpy 2, the webcam and
RealSense paths, two real machines (latency over your network), and a real grasp and lift of the cylinder (finger friction and
servo strength are untuned; the cylinder is easy to knock over).
