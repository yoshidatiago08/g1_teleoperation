# g1_teleoperation

Camera-based teleoperation of the Unitree G1, split across two machines:

```
 NOTEBOOK                                        VM (GPU)
 ┌──────────────────────────┐   Zenoh link    ┌──────────────────────────────┐
 │ RealSense ──► compress ──┼──── color+depth ─►  decompress ─► wrist tracking │
 │                          │                 │                    │ IK        │
 │ RViz + G1 model ◄────────┼─ TF, joints, ───┤◄───────────────────┘           │
 │ tracking overlay         │  overlay image  │                              │
 └──────────────────────────┘                 └──────────────────────────────┘
```

Only compressed images cross the network (~1-4 MB/s instead of ~46 MB/s raw). The notebook
needs Docker, the RealSense and a screen. No ROS install and no GPU.

> Status: right-wrist and right-elbow tracking, rest-pose calibration and right-arm IK, shown in
> RViz. Nothing here commands a physical robot. Hand orientation and fingers are not tracked.

## Run it

**VM** (this must be reachable from the notebook on port 7447):
```bash
docker compose --profile vm up --build
```

**Notebook:**
```bash
cp .env.example .env        # set TELEOP_PEER to the VM's Tailscale IP, or its LAN IP
xhost +local:docker         # lets the container open RViz on your screen
docker compose --profile notebook up --build
```

Test the link without a camera: set `FAKE_CAMERA=true` in `.env`.

Works the same over Tailscale, a LAN or a VPN: only `TELEOP_PEER` changes. After pulling new
code, run `up --build` again on the machine you updated.

## Calibration: the rest pose

Every time the VM side starts, the robot is held at its own rest pose and the camera overlay shows
**`CALIBRATING`**. Tracking only begins once you have held the rest pose for about 3 seconds.

**How to do it**
1. Stand 2 to 2.5 m from the camera, facing it, with your whole body in view.
2. Let your **right arm hang relaxed at your side**, not stiff and not bent.
3. **Hold still** until the overlay changes to **`TRACKING (calibrated)`**. The robot's arm then
   starts following you.

The overlay tells you what is missing: `stand in view of the camera`, `step back: shoulder, elbow and
wrist must all be visible`, `let your right arm hang relaxed at your side`, `hold still`, or a
countdown (`hold still... 2.1 s`). Moving during the hold restarts the countdown.

**What it does.** It measures your arm length (shoulder to elbow to wrist) over those seconds and sets
the scale `robot reach / your arm length`, so your full reach maps to the robot's full reach. Doing
it in a deliberate pose gives a clean measurement; without it, the scale was latched whenever the
samples happened to agree, which could be at a bad moment. The mapping is anchored shoulder to
shoulder: a target is your wrist's position relative to your right shoulder, scaled, placed relative
to the robot's right shoulder.

**Check it.** In the rest pose the robot's arm should hang down at its side, as straight as the G1's
elbow allows. If it does not, the camera angle or the depth is off: see "Camera placement" below.

**Calibrate again** (new person, camera moved, tracking looks scaled wrong) without restarting:
```bash
docker compose --profile vm exec tracking bash -c \
  'source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && ros2 service call /calibrate std_srvs/srv/Trigger'
```
The robot returns to rest and the overlay shows `CALIBRATING` again. Restarting the VM side does the
same.

**Skip it** by setting `CALIBRATE=false` in the VM's `.env` (the arm length is then estimated while you
move, as before).

## Settings (VM, in `.env`)

| Variable | Default | What it does |
|---|---|---|
| `CALIBRATE` | `true` | Require the rest-pose calibration before tracking. |
| `ELBOW_WEIGHT` | `0.01` | How strongly your forearm direction shapes the robot's elbow posture. `0` = follow only the wrist position; try `0.05` for stronger elbow following. |

Restart the VM side after changing them. On the notebook, `.env` holds `TELEOP_PEER` and `FAKE_CAMERA`.

## How the tracking works

1. **MediaPipe Pose** finds the right shoulder, elbow and wrist in the colour image (and the hips and
   ankles, used to find which way is up).
2. The **RealSense depth** at those pixels turns them into 3D points.
3. A **body frame** is built: its origin is always your right shoulder, its axes come from your shoulders
   and (if visible) hips and ankles, so the targets do not depend on where the camera is.
4. The wrist (and elbow) position in that frame is scaled by the calibration and published as
   `/wrist_pose` (and `/elbow_pose`), in the robot's `torso_link` frame.
5. The **IK node** finds the 7 right-arm joint angles that put the robot's wrist on the target. The
   elbow only chooses the posture: the robot's forearm is turned to point where yours does.
6. The joint angles go back to the notebook, where RViz draws the robot.

The camera overlay draws the right shoulder (cyan `S`), elbow (orange `E`) and wrist (green), the body axes
at the shoulder, and a status line (`body axes: ...  elbow: ok/LOST`).

## Camera placement

- **2 to 2.5 m away**, at about chest or hip height, level, facing you head-on. Your whole body should
  fit: the hips and ankles are used to find "up".
- Keep your **right arm visible**. Reaching straight at the camera hides the shoulder and elbow behind
  the arm; the tracker then holds the last body frame and the IK falls back to wrist-only. A camera a
  little to the side (30 to 45 degrees) helps.
- Even light, no window behind you, no direct sun. Fitted, textured clothes give better depth. Stand at
  least half a metre in front of the wall behind you.
- Use a **USB 3 port and cable**. On USB 2 the RealSense drops and corrupts frames (`lsusb -t` should
  show the camera at `5000M` or more, not `480M`).

## Layout
| Path | What |
|---|---|
| `compose.yml` | the `vm` and `notebook` profiles |
| `docker/Dockerfile` | one image per role: `teleop`, `camera`, `viz` (ROS 2 Jazzy) |
| `ros_ws/src/g1_teleop/` | the ROS package: detector, IK, fake camera, launch files, URDF |

## Topics that cross the link
| Direction | Topic |
|---|---|
| notebook → VM | `/link/color/compressed`, `/link/depth/compressedDepth`, `/camera/camera/color/camera_info` |
| VM → notebook | `/tf`, `/wrist_pose`, `/elbow_pose`, `/g1_visualization/joint_states`, `/link/debug/compressed` |

## Known limits
- One camera in front of you: occlusion when you reach toward it, and the elbow is the noisiest point.
- Only the right arm. Hand orientation and fingers are not tracked.
- The G1's elbow cannot fully straighten, so a perfectly straight hanging arm is approximated.
- The calibration is only as good as the depth: a noisy or corrupted depth stream (for example from a USB 2
  link) makes it slow to finish or inaccurate.
