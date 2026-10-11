# Isaac Sim (unitree_sim_isaaclab)

`unitree_sim_isaaclab` listens for the real robot's DDS messages, so `g1tele-robot --set robot.backend=dds`
drives it with no simulator-specific code. Topics (from its source): `rt/lowcmd` / `rt/lowstate` (G1 29-DoF,
CRC checked), `rt/inspire/cmd` / `rt/inspire/state` (12 values in [0, 1], 1 = open, right hand first),
`rt/sim_state`. DDS domain **1** (the real robot uses 0).

## Steps (robot host)

1. Install and start the simulator following its README, with the Inspire hands enabled, e.g.
   ```bash
   python sim_main.py --device cpu --enable_cameras \
     --task Isaac-PickPlace-RedBlock-G129-Inspire-Joint --enable_inspire_dds --robot_type g129
   ```
   (`...-Joint` tasks hold the base and take arm/hand commands; `...-Wholebody` tasks also walk.)
2. Install the SDK in the environment that runs `g1tele-robot` (this is separate from Isaac's Python,
   which is why DDS is the link):
   ```bash
   pip install cyclonedds
   git clone https://github.com/unitreerobotics/unitree_sdk2_python && pip install -e unitree_sdk2_python
   g1tele-doctor robot
   ```
3. Run the robot process, on the same machine and network interface as the simulator:
   ```bash
   g1tele-robot --set robot.backend=dds --set robot.dds_interface=lo
   ```
   It waits (up to 15 s) for `rt/lowstate`, starts from the robot's current pose and commands arms (and
   waist) at 250 Hz. The simulator steps its control at `--step_hz` (100 by default).
4. On the operator machine: `g1tele-perception --set link.host=<robot host>`.

DDS discovery is multicast and does not cross Tailscale/VPNs, which is why only the keypoint stream
travels between machines and `g1tele-robot` runs on the simulator's host.

## Recording and replay

Episodes use the `xr_teleoperate` schema, so the simulator's `--replay_data` / `--generate_data` tools (re-render with other lighting or
cameras) can read them. Images from Isaac's cameras are not recorded by `g1tele` yet (the simulator publishes them on its own ZMQ image server): either record them with
the simulator's tooling or ask for an image subscriber in `g1tele-robot`.

## Real robot

`--set robot.target=real --set robot.dds_domain=0 --set robot.dds_interface=<nic> --confirm-real`
switches to `rt/arm_sdk` (arms and waist only, the robot's own controller keeps balance) with a
soft-start speed ramp. Not tested on hardware: clear the area and keep the stop in reach.
