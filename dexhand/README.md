# dexhand: precise hand manipulation with dex-retargeting

Your hand in front of a camera -> the G1's **right arm** and robot hand (Inspire or Unitree Dex5-1, your choice) in MuJoCo (with a table and the red cylinder) and RViz. Fingers are
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
The status line says what the robot is doing: `PRESS C TO CALIBRATE`, `CALIBRATING`, `IMITATING`, `PAUSED`, `HAND LOST (holding)`,
`HAND UNRELIABLE (holding)` (the hand model cannot explain what the camera sees), `OUT OF REACH (arm at its limit)`.

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

**Calibrate first (press `c`):** hold your hand with the **palm facing the camera, fingers up, fingers stretched and slightly apart**, still,
for about 1.5 s (the VM then needs another 1-2 s to measure your hand). The window says `calibrated; hand 18.5 cm (...), finger scale 1.2`.
This also measures your hand (below). From then on the robot hand points where yours points: fingers at the floor = fingers at the floor. Calibration removes the
camera's tilt (if it has no IMU) and your heading, so pick and place feels aligned. Position is relative and scaled; orientation is absolute at first and relative after a pause (below).
A RealSense with an IMU (D435i/D455) levels the camera by itself from gravity, so calibration then only has to fix the heading; a plain
webcam has no IMU and relies on the calibration (or mount it level).

**Settings live on the notebook:** `MOTION_SCALE` and `ROTATION_SCALE` in the notebook's `.env` are sent to the VM (and the `+`/`-` keys
change them live). Only `FINGER_SCALE`, `TRACKING`, `HAND_LENGTH_MM` and `VM_SOURCE` are read by the VM.

**No rebuild for code changes:** `src/` and `assets/` are mounted into the containers from your checkout. After `git pull`, just restart
(`docker compose --profile vm up`, no `--build`). Rebuild only when something in `docker/` changes.

## Your hand as a model (`TRACKING=model`, the default)

MediaPipe finds the landmarks in the image well; what it estimates badly is their **depth**, above all for landmarks hidden behind the
rest of the hand (fingers pointing at the camera, a fist). The raw pipeline used that depth for the palm orientation, the fingertip vectors and the
hand's distance. Now the depth does not come from MediaPipe:

- **At the calibration** (stretched hand, palm to the camera, nothing hidden) the VM measures *your* hand: bone lengths and the shape of the palm
  (`hand_model.HandShape`). The metric scale comes from the RealSense depth at the wrist; with a webcam set `HAND_LENGTH_MM` (wrist -> middle
  fingertip, hand stretched, with a ruler), else MediaPipe's own average-hand scale is used. The robot's finger length divided by yours also sets
  `FINGER_SCALE=auto` (a fixed 1.15 before).
- **Every frame** a model of that hand (wrist pose + 20 finger joint angles with joint limits) is moved until its perspective projection lands on
  MediaPipe's *2D* landmarks (`hand_model.HandFit`). Known metric bones + perspective give the distance from the apparent size of the parts of the
  hand; the rigid palm (wrist, thumb base, 4 knuckles) gives the palm orientation without any finger; the joints give the hidden fingertips.
- MediaPipe's z is only a weak tie-breaker (a flat palm looks the same tilted either way) and only for landmarks that are **visible** in the model's
  current pose (`hand_model.visibility`: bones are capsules, the palm is triangles; a landmark behind one is hidden and its z is ignored).
- The fit error tells when it is wrong: too large (or a depth that never agrees) and the robot holds (`HAND UNRELIABLE (holding)`), and the model
  searches for the pose again from several starting orientations (24 axis-aligned ones plus MediaPipe's own).
- A RealSense depth reading is used only if the model sees the wrist (a hidden wrist reads a finger or the wall behind it).

On synthetic hands (2D noise 1.5 px, depth noise 1 cm on visible and 5 cm on hidden landmarks; `tests/test_hand_model.py`) the model, against the
raw landmarks: palm orientation error 3-7 degrees instead of 10-90 for fingers pointing at the camera or a closed hand, distance error 1-3 cm instead
of up to 36 cm, fingertip error 5-15 mm instead of 20-140 mm. A flat open palm facing the camera is the same as before (about 10 degrees of tilt
error): a flat palm's tilt is invisible in 2D and only MediaPipe's z tells it. **Not tested on a real hand yet.** `TRACKING=raw` goes back to the old
behaviour.

## Jitter: where it comes from and the smoother

MediaPipe's landmarks shake by a few pixels from frame to frame, and in some views the shake turns into a large pose error: a flat palm facing
the camera hardly changes the picture when it tilts, and the distance to the camera is always the weak direction. So the hand fit also returns, for
every frame, how uncertain the palm rotation and the wrist position are (`FitResult.cov_rot`, `cov_pos`; the window shows `palm unc N deg`, and
`(weak view)` above 10 deg). `smoothing.py` is a pair of Kalman filters (rotation, position) that use these as the measurement noise, so a direction the
image determines well follows at once and a weakly determined one follows the motion model. The process noise adapts: small while the hand is
steady (noise averages out, jitter drops), large as soon as the measurements keep pulling in one direction (a real movement, little lag).
The smoothing has two stages, both switched by `SMOOTHING`:
1. **MediaPipe's landmarks, before anything is computed from them** (`LandmarkFilter`): a One Euro filter on every landmark (pixels and MediaPipe's
   3D), strong while the landmark is steady and opening up with its speed (in hand lengths per second, so the distance does not matter), plus
   spike rejection: a landmark that jumps far relative to the rest of the hand in a single frame is not believed until the next frame confirms it.
   The hand fit, the fingertip vectors and the wrist all start from these steadier landmarks.
2. **The pose, weighted by uncertainty** (`PoseSmoother`, above): what the image still leaves undetermined after stage 1.
`SMOOTHING=off` turns both off to compare.

On the synthetic hand with MediaPipe-like noise the landmark filter halves the 2D jitter of a steady hand and removes single-frame glitches, and together with the pose
smoother the palm-rotation jitter drops by about 85% for a steady hand at any viewing angle; the lag is
small (the rotation error in fast turns grows by about 1.5 deg); a single flipped frame is ignored and a real jump is followed after 6 frames.
**Not measured on real MediaPipe landmarks yet.** Record a take with the bad angles (`RECORD=recordings/take1.npz`, then
`python tools/analyse_recording.py recordings/take1.npz --plot take1.png`): the report prints the jitter of raw MediaPipe, of the model and of the
smoothed pose, by palm angle to the camera and by finger angle to the camera, the share of frames where the image leaves the rotation poorly
determined, and the smoother's lag.

## Which robot hand: `HAND_MODEL`

| `HAND_MODEL` | Hand | Actuators | Notes |
|---|---|---|---|
| `inspire` (default) | Inspire | 6 (thumb 2, other fingers 1 each; the rest follow) | what the G1 in the main project carries |
| `dex5` | Unitree Dex5-1 | 16 (thumb 4; index, middle, ring, pinky: abduction + 2 flexions each) | the 4 distal joints follow the middle ones |

Set it in `.env` on **both** machines (like `HAND` and `MODE`) and restart, no rebuild: `HAND_MODEL=dex5`. The notebook prints a warning if the
VM runs the other hand (RViz would then show the wrong one). `--hand-model dex5` does the same on the command line.

- Everything above the hand is identical: the hand tracking, the hand frame, the scaled wrist target, CuRobo, the table. The Dex5-1 is mounted on the
  G1's wrist **turned so its hand frame is where the Inspire's was**, so the same operator pose gives the same robot pose; the wrist point is unchanged.
- The finger scale (`FINGER_SCALE=auto`) is measured against the chosen hand: the Dex5-1's fingers are shorter (about 0.17 m wrist to fingertip vs 0.21 m),
  so auto gives roughly 0.9 instead of 1.2.
- Assets come from Unitree's [unitree_ros](https://github.com/unitreerobotics/unitree_ros) (`dexterous_hand_description/dex5_1`), renamed to the Inspire
  convention (`R_hand_base_link`, `R_index_mcp_joint` ...) by `tools/make_dex5_assets.py`; the CuRobo config comes from
  `tools/make_curobo_config.py --hand-model dex5`. Outputs are committed.
- **Assumptions to check on the real thing:** (1) the passive distal joint follows the middle one with ratio 0.83 (the ratio of their joint ranges; Unitree
  does not give the linkage in the URDF); (2) the Dex5-1 is sold for the H1/H1-2, so its mount on the G1 wrist is geometric (same wrist point and hand
  frame as the Inspire), not Unitree's flange; (3) the retargeting matches only the 5 wrist -> fingertip vectors, so with 16 joints the finger spread and
  the thumb's 4 joints are chosen by dex-retargeting's regularisation, not by extra constraints.

## The arm (default mode)

- The scaled hand target (position and orientation of the hand base link, the same frame dex-retargeting uses) goes to CuRobo, which returns the
  7 right-arm joint angles (about 4 ms per frame on an RTX A2000; the first solve starts from the relaxed pose and walks to the home pose).
  Joint speed is limited to 3 rad/s, so the arm never jumps. MuJoCo moves the arm with position servos.
- **Reach:** the hand base reaches about **0.35 m** in front of the pelvis at pick height, from far right to roughly the body's centre line.
  A cylinder at x = 0.38 is grasped with the wrist near x = 0.2, inside the reach. Beyond the reach the arm goes to the nearest pose it can reach (the target is pulled back to the edge, so moving your hand back answers at once) and the window says
  `OUT OF REACH (arm at its limit)`; just move your hand back.
- **The table is avoided** (the IK keeps the arm and hand spheres out of it); **the cylinder is not**, it is what you pick up.
- Collision spheres for the pelvis, waist, torso, head and right arm come from CuRobo's own G1 configuration; the hand's (Inspire or Dex5-1) are fitted to its
  meshes (`tools/make_curobo_config.py`). The robot is the G1 without legs (`tools/make_g1_upper_urdf.py`).
- CuRobo's "differential IK" mode (`run_optimizer=False`) does not move the arm in v0.8.0 (the LM seed solver only runs with the optimizer), so
  each frame is a one-seed IK warm-started from the previous solution, with a multi-seed fallback after 5 failed frames.

## How it works

```
frames ─► MediaPipe ─► wrist -> fingertip vectors ─► dex-retargeting (URDF + mimic joints) ─► Inspire (6) or Dex5-1 (16) joints ─► MuJoCo servos
                  └─► wrist pixel + depth ─► operator axes ─► MotionScaler (scale, clutch) ─► weld target ─► MuJoCo hand
```

- `src/dex_hand/` (pure Python, no ROS): `frames.py` hand frame and axes, `motion.py` scaler/clutch, `retarget.py` dex-retargeting
  wrapper, `pipeline.py` glue, `arm_ik.py` the CuRobo arm IK, `mujoco_arm_scene.py` the G1 arm scene, `mujoco_scene.py` the floating-hand scene, `vm.py` the VM app, `notebook.py` + `capture.py` the notebook app,
  `hand_model.py` your hand as a model fitted to the 2D landmarks, `link.py` the ZeroMQ link, `ros_bridge.py` RViz (notebook only, optional).
- Two different scales: `FINGER_SCALE` (dex's `scaling_factor`, the robot's finger length vs yours, ~1.2; `auto` measures it at the calibration) and `MOTION_SCALE` (hand
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
The hand model is verified on synthetic hands only (numbers above, `tests/test_hand_model.py`, and in the VM with the made-up hand: the calibration measures it
and the status reaches `IMITATING` with `depth_from: hand model`); its speed is about 14 ms per frame on one CPU core (the first frame after a calibration or a lost hand
takes about 0.6 s: it searches the starting pose).
Not verified: MediaPipe's real 2D landmarks under occlusion (the model rests on them being good), the real MediaPipe z noise (the tilt of an open palm depends on it), the RViz window and the notebook window (no display there), MediaPipe on a real hand under numpy 2, the webcam and RealSense paths,
two real machines (latency over your network), and a real grasp and lift of the cylinder (finger friction and servo strength are untuned; the cylinder
is easy to knock over; the arm's reach is short, see above).
