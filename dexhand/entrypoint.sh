#!/bin/bash
# Load ROS and the built workspace, then run whatever command the container was given.
set -e
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /ws/install/setup.bash
exec "$@"
