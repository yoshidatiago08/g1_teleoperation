# dexhand: precise hand manipulation with dex-retargeting

Your hand in front of a camera -> the G1's **right arm** and Inspire hand in MuJoCo (with a table and the red cylinder) and RViz. Fingers are
retargeted with [dex-retargeting](https://github.com/dexsuite/dex-retargeting); the hand's position is **scaled and clutched**, so a
large motion of yours is a small, precise motion of the robot hand, and the arm follows through GPU inverse kinematics
([CuRobo](https://github.com/NVlabs/curobo) v0.8.0 / cuRoboV2) that keeps the arm clear of the table. The waist, legs and left arm are static.
`MODE=floating` is the fallback: a floating hand, no arm (runs without a GPU).

Same split as the main project: **the VM does everything heavy, the notebook only the camera and the display.**
This directory is self-contained (own Docker images, own compose); it shares nothing with the whole-body `g1tele` images.

```
 NOTEBOOK (light, no GPU)                               VM (heavy)
 ┌─────────────────────────────┐   JPEG frames        ┌──────────────────────────────────────────┐
 │ camera ─► JPEG ─────────────┼────────────────────► │ MediaPipe ─► dex-retargeting ─► scaler   │
 │                             │   (+ small depth)    │              ─► CuRobo arm IK ─► │       │
 │ window: preview + simulator │ ◄────────────────────┤ MuJoCo physics + offscreen render ◄──────┘
 │ RViz (G1, table, cylinder)  │   state (1 kB) +     │                                          │
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
The VM needs an NVIDIA GPU and the NVIDIA container toolkit (arm IK and the simulator video); the first start compiles GPU kernels for about a
minute (cached in a Docker volume afterwards). Without a GPU use `MODE=floating` (on both machines) and `MUJOCO_GL=osmesa`.
Ports 5601-5603 must be reachable from the notebook to the VM.

Sit 0.5-0.8 m from the camera (MediaPipe needs the hand to be at least ~100 px wide; at 2 m it is ~30 px), the camera at hand
height and facing you (level, unless it is a RealSense with an IMU), hand fully visible. Then press `c` once to calibrate (below).

## What you see

A window with your camera (the VM draws the tracked hand on it) next to the simulator view, plus RViz. In MuJoCo the hand collides:
it pushes the cylinder and cannot go through the table. Text shows the state, the latency and `cylinder -> target X cm, lifted Y cm`.

| In RViz | Meaning |
|---|---|
| The G1 (arm mode) or the Inspire hand (floating) | the **simulated** robot (what the IK and physics allowed) |
| `hand_target` axes | the scaled target the simulated hand is pulled towards |
| Cyan skeleton + `operator_hand` axes | your tracked hand at **scale 1** from the same starting point, to compare |

Keys in the window: **`c` calibrate**, **`space` pause / resume imitation**, `a` re-align orientation, `+` / `-` motion scale x1.25 / x0.8, `o` cylinder back, `q` quit.
The status line says what the robot is doing: `PRESS C TO CALIBRATE`, `CALIBRATING`, `IMITATING`, `PAUSED`, `HAND LOST (holding)`, `OUT OF REACH (arm holding)`.

- **Nothing moves until the first calibration is done**, then imitation starts by itself.
- **Pause (`space`)** stops all imitation, wrist and fingers: the simulated hand stays exactly where it is. Use it when your hand is at the edge of
  what the camera sees: pause, bring your hand back to the middle, resume (`space` again): the robot continues from where it was, moving by
  how far your hand moves from there on (no jump). Losing the hand does the same automatically.
- **Translation follows your hand's own axes.** Moving your hand along its fingers moves the robot along *its* fingers, whatever the robot's
  orientation (the mapping is fixed when you resume, from your hand's orientation averaged over a few frames, so nothing accumulates noise).
  Right after calibration this is the same as plain world directions; it matters once the robot is turned for a side grasp: keep your hand
  flat in front of the camera and move it forward/back/left/right/up/down, and the robot moves that way relative to its own hand.
- **Orientation accumulates across pauses.** The first imitation after a calibration is absolute (fingers at the floor = fingers at the floor).
  After a pause (or a lost hand) the robot keeps the orientation it has and applies only how much you rotate your hand from then on, about
  the world axes. So for a side grasp: turn your hand a little while the camera still sees it, pause, flatten the hand, resume, turn a
  little more. The fingers always follow your finger pose exactly. `a` re-aligns the robot to your hand's absolute orientation again.

**Calibrate first (press `c`):** hold your hand with the **palm facing the camera, fingers up**, still, for about 1.5 s. The window says
`calibrated`. From then on the robot hand points where yours points: fingers at the floor = fingers at the floor. Calibration removes the
camera's tilt (if it has no IMU) and your heading, so pick and place feels aligned. Position is relative and scaled; orientation is absolute at first and relative after a pause (below).
A RealSense with an IMU (D435i/D455) levels the camera by itself from gravity, so calibration then only has to fix the heading; a plain
webcam has no IMU and relies on the calibration (or mount it level).

**Settings live on the notebook:** `MOTION_SCALE` and `ROTATION_SCALE` in the notebook's `.env` are sent to the VM (and the `+`/`-` keys
change them live). Only `FINGER_SCALE` and `VM_SOURCE` are read by the VM.

**No rebuild for code changes:** `src/` and `assets/` are mounted into the containers from your checkout. After `git pull`, just restart
(`docker compose --profile vm up`, no `--build`). Rebuild only when something in `docker/` changes.

## The arm (default mode)

- The scaled hand target (position and orientation of the hand base link, the same frame dex-retargeting uses) goes to CuRobo, which returns the
  7 right-arm joint angles (about 4 ms per frame on an RTX A2000; the first solve starts from the relaxed pose and walks to the home pose).
  Joint speed is limited to 3 rad/s, so the arm never jumps. MuJoCo moves the arm with position servos.
- **Reach:** the hand base reaches about **0.35 m** in front of the pelvis at pick height, from far right to roughly the body's centre line.
  A cylinder at x = 0.38 is grasped with the wrist near x = 0.2, inside the reach. Beyond the reach the arm holds its last pose and the window says
  `OUT OF REACH (arm holding)`; pause, move your hand back, resume (the scale and the pause are there for exactly this).
- **The table is avoided** (the IK keeps the arm and hand spheres out of it); **the cylinder is not**, it is what you pick up.
- Collision spheres for the pelvis, waist, torso, head and right arm come from CuRobo's own G1 configuration; the Inspire hand's are fitted to its
  meshes (`tools/make_curobo_config.py`). The robot is the G1 without legs (`tools/make_g1_upper_urdf.py`).
- CuRobo's "differential IK" mode (`run_optimizer=False`) does not move the arm in v0.8.0 (the LM seed solver only runs with the optimizer), so
  each frame is a one-seed IK warm-started from the previous solution, with a multi-seed fallback after 5 failed frames.

## How it works

```
frames ─► MediaPipe ─► wrist -> fingertip vectors ─► dex-retargeting (URDF + mimic joints) ─► 6 Inspire joints ─► MuJoCo servos
                  └─► wrist pixel + depth ─► operator axes ─► MotionScaler (scale, clutch) ─► weld target ─► MuJoCo hand
```

- `src/dex_hand/` (pure Python, no ROS): `frames.py` hand frame and axes, `motion.py` scaler/clutch, `retarget.py` dex-retargeting
  wrapper, `pipeline.py` glue, `arm_ik.py` the CuRobo arm IK, `mujoco_arm_scene.py` the G1 arm scene, `mujoco_scene.py` the floating-hand scene, `vm.py` the VM app, `notebook.py` + `capture.py` the notebook app,
  `link.py` the ZeroMQ link, `ros_bridge.py` RViz (notebook only, optional).
- Two different scales: `FINGER_SCALE` (dex's `scaling_factor`, your finger length vs the robot's, ~1.15) and `MOTION_SCALE` (hand
  travel, ~0.3). Rotation has its own `ROTATION_SCALE` (1.0 = turns like your hand).
- Reference: position is relative (`anchor + scale * M * (your hand - anchor)`, the anchor is taken when tracking starts, after the clutch or
  after the hand was lost; M maps your hand's axes onto the robot hand's, identity right after a calibration), orientation is absolute for the first imitation after a calibration (the robot hand points where yours points) and relative after a
  pause or a lost hand (the robot keeps its orientation, your rotation from then on is added; `ROTATION_SCALE` scales it; turning is
  rate-limited so nothing snaps). The operator frame is levelled by the RealSense
  IMU's gravity when there is one (`frames.gravity_aligned_axes`), and the calibration pose fixes the rest.
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

Verified (on the development machine, with a GPU but no camera and no screen): 74 tests in the VM image (physics, link, frame encoding, the whole VM
app in both modes, CuRobo IK accuracy / speed / joint-speed limit / reach limit / table avoidance, MuJoCo and CuRobo agreeing on where the hand
is for the same joints) and 30 in the notebook image (ROS bridge); the VM and notebook containers running against each other in arm mode with the
synthetic hand (calibration, `IMITATING`, 41 robot joints streamed, `/joint_states` and TF of the whole G1 in ROS); the arm driven by the IK to the
cylinder in MuJoCo, where pushing it moves it.
Not verified: the RViz window and the notebook window (no display there), MediaPipe on a real hand under numpy 2, the webcam and RealSense paths,
two real machines (latency over your network), and a real grasp and lift of the cylinder (finger friction and servo strength are untuned; the cylinder
is easy to knock over; the arm's reach is short, see above).
