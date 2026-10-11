"""g1tele: camera-based teleoperation of the Unitree G1 for manipulation datasets in simulation.

Layout (each layer only depends on the ones above it in this list):
    protocol / link   what crosses the network between the operator's machine and the simulator host
    perception        camera -> filtered operator state (runs where the camera is)
    retarget          operator state -> robot joint targets (runs next to the simulator)
    robot             backends that apply joint targets: kinematic, MuJoCo, Unitree DDS (Isaac / real G1)
"""
__version__ = "0.2.0"
