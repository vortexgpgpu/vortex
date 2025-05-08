#!/bin/bash

# Set environment variables with fallback default values.
# The :- means if the variable CONTAINER_IMAGE is not set or is empty, then use the default value vortex.sif
export GIT_REPO_PATH=${GIT_REPO_PATH:-~/USERSCRATCH/vortex}
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-vortex.sif}
export DEVNULL_BIND=${DEVNULL_BIND:-~/USERSCRATCH/devnull}
export VORTEX_TOOLCHAIN_PATH=${VORTEX_TOOLCHAIN_PATH:-~/USERSCRATCH/tools}

# Launch the Apptainer container with the bind mount
apptainer shell --fakeroot --cleanenv --writable-tmpfs \
  --bind /dev/bus/usb,/sys/bus/pci \
  --bind /projects:/projects \
  --bind /lib/firmware:/lib/firmware \
  --bind /opt/xilinx/:/opt/xilinx/ \
  --bind /tools:/tools \
  --bind /netscratch:/netscratch \
  --bind "$VORTEX_TOOLCHAIN_PATH":/home/tools \
  --bind "$GIT_REPO_PATH":/home/vortex \
  "$CONTAINER_IMAGE"