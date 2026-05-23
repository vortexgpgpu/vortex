# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Vulkan test suite -- exercises the vortexpipe Gallium driver
# through Mesa's lavapipe ICD. See tests/vulkan/common.mk.

ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

all:
	$(MAKE) -C compute
	$(MAKE) -C cflow
	$(MAKE) -C shmem
	$(MAKE) -C triangle
	$(MAKE) -C depth
	$(MAKE) -C textured
	$(MAKE) -C draw3d
	$(MAKE) -C raytrace

run-simx:
	$(MAKE) -C compute run-simx
	$(MAKE) -C cflow run-simx
	$(MAKE) -C shmem run-simx
	$(MAKE) -C triangle run-simx
	$(MAKE) -C depth run-simx
	$(MAKE) -C textured run-simx
	$(MAKE) -C draw3d run-simx
	$(MAKE) -C raytrace run-simx

clean:
	$(MAKE) -C compute clean
	$(MAKE) -C cflow clean
	$(MAKE) -C shmem clean
	$(MAKE) -C triangle clean
	$(MAKE) -C depth clean
	$(MAKE) -C textured clean
	$(MAKE) -C draw3d clean
	$(MAKE) -C raytrace clean

.PHONY: all run-simx clean
