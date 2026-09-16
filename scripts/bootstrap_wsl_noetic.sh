#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg lsb-release

# This machine already has a working ROS mirror configured. Avoid downloading
# the key from raw.githubusercontent.com, which is unreliable on this network.
if ! apt-cache show ros-noetic-catkin >/dev/null 2>&1; then
  echo 'No ROS Noetic apt source is configured.' >&2
  exit 2
fi

apt-get update
apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  libceres-dev \
  libeigen3-dev \
  libopencv-dev \
  python3-catkin-tools \
  python3-pip \
  python3-rosdep \
  ros-noetic-catkin \
  ros-noetic-cv-bridge \
  ros-noetic-image-transport \
  ros-noetic-message-filters \
  ros-noetic-rosbag \
  ros-noetic-roslaunch \
  ros-noetic-tf

echo 'ROS Noetic dependencies installed.'
