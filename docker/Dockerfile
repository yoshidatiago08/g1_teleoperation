# One Dockerfile, three roles. Build a role with `--target <role>` (compose does this for you):
#   teleop : runs on the VM      - tracking + IK
#   camera : runs on the notebook - RealSense driver + compression
#   viz    : runs on the notebook - RViz
ARG ROS_DISTRO=jazzy

# ---- base: ROS + Zenoh middleware + image compression, shared by every role ----
FROM ros:${ROS_DISTRO}-ros-base AS base
ARG ROS_DISTRO
ENV DEBIAN_FRONTEND=noninteractive \
    RMW_IMPLEMENTATION=rmw_zenoh_cpp
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-rmw-zenoh-cpp \
      ros-${ROS_DISTRO}-image-transport \
      ros-${ROS_DISTRO}-image-transport-plugins \
      ros-${ROS_DISTRO}-cv-bridge \
      ros-${ROS_DISTRO}-message-filters \
      ros-${ROS_DISTRO}-tf2-ros \
      ros-${ROS_DISTRO}-robot-state-publisher \
      python3-numpy \
    && rm -rf /var/lib/apt/lists/*
COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

# ---- build: compile our ROS package once, reuse the result in every role ----
FROM base AS build
RUN apt-get update && apt-get install -y --no-install-recommends python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /ws
COPY ros_ws/src src
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && colcon build --packages-select g1_teleop

# ---- teleop (VM) ----
FROM base AS teleop
# mediapipe is pinned: newer releases removed the `mp.solutions.pose` API the detector uses.
# --break-system-packages is fine here: the container exists only to run this project.
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir --break-system-packages "mediapipe==0.10.14" "numpy<2"
COPY --from=build /ws/install /ws/install

# ---- camera (notebook) ----
FROM base AS camera
ARG ROS_DISTRO
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-realsense2-camera usbutils \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /ws/install /ws/install

# ---- viz (notebook) ----
FROM base AS viz
ARG ROS_DISTRO
RUN apt-get update && apt-get install -y --no-install-recommends ros-${ROS_DISTRO}-rviz2 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /ws/install /ws/install
