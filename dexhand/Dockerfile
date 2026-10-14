# Self-contained image for the dex-retargeting hand viewer. Shares nothing with the main g1tele images.
ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-base
ARG ROS_DISTRO
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/pydeps \
    LIBGL_ALWAYS_SOFTWARE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-rviz2 ros-${ROS_DISTRO}-robot-state-publisher ros-${ROS_DISTRO}-tf2-ros \
      python3-pip python3-colcon-common-extensions libgl1 libglib2.0-0 libxkbcommon-x11-0 usbutils \
    && rm -rf /var/lib/apt/lists/*

# Python deps live in /opt/pydeps (first on PYTHONPATH), so numpy 2 here never touches ROS's own packages.
#  - dex_retargeting: --no-deps because it declares numpy>=2 and we pick mediapipe's needs ourselves
#  - mediapipe 0.10.14: the last release with the mp.solutions API; --no-deps because it pins numpy<2 (runs fine on 2)
#  - torch: dex-retargeting's optimiser uses it (CPU build)
RUN pip3 install --no-cache-dir --break-system-packages --target /opt/pydeps \
      "numpy>=2" pin nlopt pytransform3d anytree lxml pyyaml pytest pyrealsense2 \
      absl-py attrs flatbuffers "protobuf<5" opencv-contrib-python matplotlib \
 && pip3 install --no-cache-dir --break-system-packages --target /opt/pydeps --no-deps \
      "mediapipe==0.10.14" "dex_retargeting==0.5.0" \
 && pip3 install --no-cache-dir --break-system-packages --target /opt/pydeps --no-deps \
      torch --index-url https://download.pytorch.org/whl/cpu \
 && pip3 install --no-cache-dir --break-system-packages --target /opt/pydeps \
      filelock sympy networkx jinja2 fsspec typing_extensions "setuptools<81"

# pin (pinocchio) installs its modules under cmeel.prefix and relies on a .pth file, which --target dirs ignore
ENV PYTHONPATH=/opt/pydeps:/opt/pydeps/cmeel.prefix/lib/python3.12/site-packages

WORKDIR /ws
COPY ws/src src
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && colcon build --packages-select dex_hand
COPY tests /app/tests
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "dex_hand", "dex_hand.launch.py"]
