#!/bin/bash
# Load ROS (for RViz), then run whatever command the container was given.
set -e
source "/opt/ros/${ROS_DISTRO}/setup.bash"
exec "$@"
