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

> Status: right-wrist tracking + position-only right-arm IK, shown in RViz. Nothing here
> commands a physical robot.

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

Works the same over Tailscale, a LAN or a VPN: only `TELEOP_PEER` changes.

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
| VM → notebook | `/tf`, `/wrist_pose`, `/g1_visualization/joint_states`, `/link/debug/compressed` |
