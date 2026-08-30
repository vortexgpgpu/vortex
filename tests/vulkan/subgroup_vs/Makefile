# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Vertex-stage subgroup test for the vortexpipe driver: a full-target triangle
# whose vertex shader reports its own subgroup size, lane id, a broadcast and a
# ballot, forwarded to the colour target and decoded on the host.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render + check the vertex stage's subgroup report

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := subgroup_vs

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

# Order matters: main.c takes the vertex module as argv[1], the fragment
# module as argv[2].
SHADERS := $(SRC_DIR)/subgroup_vs.vert $(SRC_DIR)/subgroup_vs.frag

# GL_KHR_shader_subgroup_* lower to SPIR-V 1.3 group operations, which glslc
# only emits for Vulkan 1.1 or later; its default target is Vulkan 1.0.
GLSLC := glslc --target-env=vulkan1.1

include ../common.mk
