# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# mesh -- task and mesh shaders on the device: static and run-time counts,
# lines and points, culling, workgroup memory, task payloads, per-primitive
# outputs and layers, indirect draws, the limits, task chunks and mesh windows,
# rasterizer discard and the stages' queries, each compared with the image or
# the counts the specification gives.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run every case
#
# STRICT stays at the suite default of 1: llvmpipe runs task and mesh shaders
# correctly, so a draw that fell back would pass without testing the device.
# A task or mesh workgroup runs as one CTA, and the specification's minimums
# (128 invocations, 32 KiB of task workgroup memory) need the wide core.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := mesh
WIDE := 1

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/tri.mesh $(SRC_DIR)/grid_static.mesh $(SRC_DIR)/grid_dyn.mesh \
           $(SRC_DIR)/grid_line.mesh $(SRC_DIR)/grid_point.mesh $(SRC_DIR)/grid_cull.mesh \
           $(SRC_DIR)/shared.mesh $(SRC_DIR)/task.task $(SRC_DIR)/task_fixed.task \
           $(SRC_DIR)/task.mesh $(SRC_DIR)/perprim.mesh $(SRC_DIR)/perprim.frag \
           $(SRC_DIR)/grid_draw.mesh $(SRC_DIR)/big.mesh $(SRC_DIR)/fat.mesh $(SRC_DIR)/flat.frag \
           $(SRC_DIR)/maxmem.mesh $(SRC_DIR)/partial.mesh $(SRC_DIR)/layer.mesh \
           $(SRC_DIR)/taskbig.task $(SRC_DIR)/taskbig.mesh $(SRC_DIR)/taskwin.task \
           $(SRC_DIR)/taskwin.mesh

# The shaders share mesh_common.glsl, and glslc compiles task and mesh stages
# only for a Vulkan 1.2 target or later. Ahead of common.mk's rule, which make
# would otherwise pick.
%.spv: $(SRC_DIR)/% $(SRC_DIR)/mesh_common.glsl
	$(GLSLC) --target-env=vulkan1.3 $< -o $@

include ../common.mk
