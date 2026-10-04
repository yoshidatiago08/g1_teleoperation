#!/bin/bash
# Load ROS and our workspace, then run whatever command the container was given.
set -e
source "/opt/ros/${ROS_DISTRO}/setup.bash"
[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash
exec "$@"
