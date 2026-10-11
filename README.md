# g1tele

Camera-based teleoperation of the Unitree G1 (arms and Inspire hands) for **collecting manipulation
datasets in simulation**: Isaac Sim (through `unitree_sim_isaaclab`) and MuJoCo.

One RGB-D camera looks at the operator. The robot's hands should end up where yours are, fast and
steady enough for imitation learning.

```
 OPERATOR MACHINE (camera, screen)                      ROBOT HOST (VM, simulator)
 ┌──────────────────────────────────┐   1 kB/frame     ┌─────────────────────────────────────┐
 │ RealSense ─► MediaPipe ─► depth  │ ───────────────► │ retarget (IK) ─► backend            │
 │   lift, bone fusion, filters,    │   ZeroMQ         │    ├─ MuJoCo (in process)           │
 │   calibration, hands             │                  │    └─ Unitree DDS ─► Isaac Sim / G1 │
 │                                  │ ◄─────────────── │ episode recorder                    │
 │ window: overlay + simulator view │  status + video  └─────────────────────────────────────┘
 └──────────────────────────────────┘
```

Only the operator's *keypoints* cross the network (about 1 kB per frame), never images. The robot host
does the inverse kinematics next to the simulator, so the feedback loop stays local.

> **Status.** The whole pipeline is implemented and tested end to end on a scripted synthetic operator
> (accuracy, jitter, latency, occlusion), and the MediaPipe/RealSense/DDS code paths run, but the numbers
> below come from that synthetic world. **First thing to do on real hardware: `g1tele-doctor operator`,
> then record a session and tune with `g1tele-bench`** (see [docs/tuning.md](docs/tuning.md)).

## Quick start (two commands, like before)

**On both machines:** `git clone`, then `cp env.example .env`.

**Robot host (the VM running the simulator):**
```bash
docker compose --profile robot up --build
```

**Operator machine (camera + screen):** edit `.env` and set `ROBOT_HOST` to the VM's Tailscale/LAN IP, then
```bash
xhost +local:docker
docker compose --profile operator up --build
```

A window opens with the camera, what the tracker sees and the simulated robot. Stand 1.5-2 m from the
camera and follow the on-screen prompts (calibration, about 20 s the first time, 3 s after that): when it says `TRACKING`
the robot follows you. Keys in the window: `c` full calibration, `v` quick calibration, `o` put the cylinder back, `r` record an episode, `q` quit.

Needs Docker and, on the VM, the NVIDIA container toolkit (the robot video is rendered on the GPU; without a GPU the
robot still works, just without the video). Everything else (the camera's USB access, the window) is set up in the compose file.
No camera? set `CAMERA=synthetic` in `.env` for a scripted operator. Other options (Isaac Sim backend, operator name, extra settings) are
commented in `env.example`.

## Without Docker

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[perception]"        # operator machine; the robot host needs ".[mujoco,dds]"
pip uninstall -y opencv-python-headless && pip install opencv-python   # only if you want the live window
g1tele-doctor operator                # checks the camera (USB 3, IMU), MediaPipe, display

g1tele-robot --set robot.render=true                    # robot host
g1tele-perception --set link.host=<VM IP>               # operator machine
```
All settings are in one YAML ([configs/default.yaml](configs/default.yaml), every default documented);
copy it and pass `--config mine.yaml`, or override with `--set section.key=value`. For Isaac Sim see
[docs/isaac.md](docs/isaac.md).

## Calibration

The first run walks you through a few poses (each is held for about 2 s; the overlay says what to do and
what is wrong):

1. **rest**: arms hanging relaxed
2. **side**: arms straight out to the sides (T-pose)
3. **open hand**, then **fist**: this person's finger range

It learns your **bone lengths** (checked for consistency between poses and between arms), **"up"** (from
the camera's IMU if it has one, else from the floor/desk in view, else from your arms), the body frame and
your finger range. This is saved to `profiles/<operator>.json`; the next session only repeats the rest pose
(`--full-calibration` redoes everything). Use `--operator name` for several people on one machine.

**Calibration is checked all the time.** A finished calibration is validated (plausible arm lengths, left/right agreement,
facing the camera...) and rejected and redone if it fails; it is never saved in that case. While tracking, a monitor compares the measured
arm lengths with the calibrated ones and the camera's "up" (IMU or floor) with the calibrated one: if they stop matching for a few
seconds (bad calibration, camera bumped, another person) it recalibrates by itself and says why on screen. No more restarting the
container. A yellow/red `WARNING` line means it is unsure; `v` redoes the rest pose.

## Checking your control: the cylinder (MuJoCo)

The MuJoCo scene has a table, a red cylinder and a green target disc. The video shows the robot's **first-person camera** and an upper-body view side by side
(`robot.cameras=[head]` for first person only; the window shows the first-person view and the upper body side by side at up to 60 fps; `robot.scene=false` removes the scene). Push, grasp or carry the cylinder to the green disc; the window shows the
distance to the target and whether it is lifted. `o` puts it back. Only the hands collide with the scene.

## Isaac Sim

[docs/isaac.md](docs/isaac.md) has the full steps. In short: run `unitree_sim_isaaclab` with
`--enable_inspire_dds` on a `...-G129-Inspire-Joint` task, then

```bash
g1tele-robot --set robot.backend=dds --set robot.dds_interface=lo
```
`unitree_sim_isaaclab` speaks the robot's own DDS protocol, so the same backend drives a real G1
(`robot.target=real`, extra confirmation required).

## Recording datasets

Press `r` in the operator window (or `link.send_event(cmd='record_start')`). Each episode is written on
the robot host to `recordings/<task>/episode_NNNN/`:

| file | content |
|---|---|
| `data.json` | same schema as unitree's `xr_teleoperate` / `unitree_sim_isaaclab` episodes: replay and data-generation tools of `unitree_sim_isaaclab` work on it |
| `operator.msgpack` | the operator's raw keypoints: `g1tele-replay` re-retargets them with other settings or another backend, no new demonstration needed |
| `quality.json` | per step: tracking status (`TRACKING`/`HELD`/`LOST`), IK error, latency: filter bad segments before training |
| `colors/` | simulator camera frames (MuJoCo backend) |

## How the precision is achieved

Measured on a scripted synthetic operator with `g1tele-bench` (limbs occluding each other, arms leaving the
frame, camera tilted 10°). The noise level matters a lot: **tuned on clean noise (1 px landmark jitter) the filters shook a
still arm by 48 mm once the noise was MediaPipe-like (4 px)**, so the defaults are now chosen at the realistic level. Real numbers
will differ again: use the same tool on your own recordings ([docs/tuning.md](docs/tuning.md)).

| MediaPipe-like noise (4 px landmarks, noisier depth) | |
|---|---|
| wrist, vs ground truth | **20 mm RMS, 38 mm p95** (includes about 33 ms of filter lag) |
| still-arm scatter | **5.5 mm** |
| operator → robot wrist, MuJoCo closed loop (tracked frames) | 26 mm p95 |
| end-to-end delay (30 fps camera, link, IK, servo) | 60-90 ms (33 ms of it is the camera frame) |
| perception time per frame | 5-6 ms, plus MediaPipe (about 13 ms full model on a laptop-class CPU) |

With clean noise the same defaults give 1.3 mm still scatter and ~22 mm RMS; the lag is the price of steadiness. The trade-off is one knob
(`perception.filters.wrist_beta`, higher = less lag, more shake).

What does the work (details in [docs/architecture.md](docs/architecture.md)):

- **Joint centre, not skin.** Depth sees the skin in front of a joint: points are moved along the ray by the limb radius, and the *near* part of a depth window is used (a median bleeds background in at limb edges).
- **Kinematic depth fusion.** MediaPipe's 2D landmarks are good, depth under an occluding limb is not. The 3D arm is solved from the 2D rays, the calibrated bone lengths and the depth readings (robust: an occluded reading is discounted, and the wide-scale scoring picks the right one of the two ray/bone-sphere intersections). An arm pointing at the camera, which hides its own elbow, is the case this fixes.
- **Arms pointing at the camera (the hard case).** The closer the forearm points to the camera, the more the hand covers the wrist and the elbow is a guess: the depth under the wrist reads up to 8 cm too near and flickers with every pixel of landmark jitter. So the elbow's trust is measured from how short the forearm looks in the image; the wrist then comes from its own ray and depth alone (the elbow can no longer move it), its depth is corrected for the hand in front of it, and its distance (noisy) is filtered separately from its direction (accurate) and harder the more it points at the camera. The elbow is then only placed where the bone lengths allow, as a *swivel hint*. In the table-reach benchmark this took the robot wrist from a 12 mm shake and 98 mm single-frame jumps to under 2 mm and 22 mm.
- **A correct "up".** Without it the whole body frame tilts: a 5° error is 45 mm at arm's length. IMU, else floor plane (RANSAC on the depth image), else the calibration poses.
- **One Euro filters** instead of an EMA: smooth when still, no lag when moving fast.
- **Bone-direction retargeting.** Each of your bones keeps its direction and takes the robot's length, so the posture is copied exactly (the elbow too) and a stretched arm is a stretched robot arm. The IK is a damped least-squares solve with analytic Jacobians: about 2 ms per arm, converging to 0.1 mm.
- **Five independent fingers**, from joint angles of the 21 hand landmarks, scaled to *your* open-hand/fist range.
- **No jumps at start or after a loss:** when tracking starts (or resumes) the arm leaves where it is and eases into your pose over about a second; during calibration or a loss it *holds* (it used to sweep to a rest pose, across the table), and only drifts slowly to rest after 3 s. The robot starts in a hanging-arms pose away from the table.
- **Failures degrade, they do not jump:** hidden shoulder held, speed gate (a wrist cannot move 5 m/s), `HELD` then `LOST` states, the robot eases to rest instead of freezing, every step carries a quality flag.

## Settings you will touch

| Setting | Default | |
|---|---|---|
| `link.host` | `127.0.0.1` | address of the robot host, from the operator machine |
| `robot.backend` | `mujoco` | `kinematic`, `mujoco` or `dds` |
| `perception.hands` | `true` | palm orientation + five fingers; `false` is faster (arms only) |
| `perception.waist` | `false` | waist from the shoulders. Off for manipulation datasets: shoulder noise would move the torso |
| `perception.arms` | `[right, left]` | track one arm for a one-handed task |
| `perception.pose_model_complexity` | `1` | `0` is about 2x faster, a little less steady |
| `perception.filters.wrist_min_cutoff` / `wrist_beta` | `1.0` / `5` | smoothness when still / lag when fast |
| `perception.depth_sigma`, `bone_sigma` | `0.02`, `0.02` | how much depth vs. bone lengths decide (smaller `bone_sigma` is more accurate but shakes with noisy landmarks) |
| `perception.fusion` | `true` | `false` = raw lifted joints, to compare against |
| `perception.pose_smoothing` | `true` | MediaPipe's own landmark smoothing |
| `camera.fps`, `camera.depth_preset` | `30`, `Default` | as in the old pipeline (640x480x30) |
| `ik.mapping` | `bone_scaled` | `uniform` or `metric` (see `retarget/retargeter.py`) |
| `ik.elbow_weight`, `ik.rot_weight` | `0.35`, `0.12` | arm posture / hand orientation vs. wrist position |

## Camera placement

- **1.5-2 m**, chest height, facing you. Closer is more precise, but the arms must stay in the frame.
- **Tilt it down 10-15°, or use a camera with an IMU (D435i/D455).** Without either, "up" comes from your arms (about 5° worse: see above).
- **Slightly to the side or above** helps with occlusion: an arm pointing straight at the camera hides its own elbow and shoulder. The fusion copes, but a camera that never sees that case is better.
- USB 3 port and cable (`g1tele-doctor operator` checks it). Even light, fitted textured clothes.

## Layout

| Path | What |
|---|---|
| `src/g1tele/protocol.py`, `link.py` | what crosses the network (`OperatorFrame`), ZeroMQ link |
| `src/g1tele/perception/` | camera, lifting, fusion, tracker, calibration, pipeline, synthetic operator |
| `src/g1tele/retarget/` | URDF reader, IK, hand mapping, `Retargeter` |
| `src/g1tele/robot/` | backends (`kinematic`, `mujoco_backend`, `unitree_dds`), `runner` |
| `src/g1tele/bench.py`, `cli/` | benchmark harness and the commands |
| `configs/default.yaml` | every setting (generated: `python -m g1tele.config`) |
| `assets/g1/` | G1 29-DoF + Inspire hands URDF and meshes |
| `tests/` | `pytest` (fast) and `pytest -m slow` (end to end, about 90 s) |
| `docs/` | architecture, tuning, Isaac Sim, adding a backend |
| `legacy/`, `ros_ws/` | the previous ROS 2 pipeline, kept for reference (see below) |

## Teammates: using it on another machine

`git clone`, the install lines above, `g1tele-doctor`, `--set link.host=<robot host>`. Nothing else is
machine-specific. To support another simulator or robot, implement one class
([docs/adding-a-backend.md](docs/adding-a-backend.md)).

## Legacy ROS 2 pipeline

The earlier version (ROS 2 Jazzy, Zenoh, RViz, images over the network) is in `ros_ws/` and
`legacy/compose.ros.yml` and still builds (`docker compose -f legacy/compose.ros.yml ...`). It is no longer
developed: the new pipeline replaces its transport, tracking, IK and visualisation.

## Known limits

- One camera: an arm pointing at it is estimated from bones and 2D, a hand behind the body is lost (`HELD`/`LOST` flags it).
- The G1's arm is shorter than a human's: the robot copies your *posture*, scaled by bone length (`ik.mapping`).
- The Inspire hand has 6 actuators for 5 fingers: the thumb has two (bend, rotation); ring and middle are independent here, the hand model's mimic joints do the rest.
- MuJoCo scene is the robot only (no table, no objects yet); Isaac tasks bring their own scenes.
- Validated on synthetic data and the MediaPipe/RealSense/MuJoCo code paths. The Unitree DDS backend is tested with a stub SDK and has **not** been run against `unitree_sim_isaaclab` or a robot yet.
