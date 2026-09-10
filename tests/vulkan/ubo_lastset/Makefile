# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Highest-descriptor-set UBO test for the vortexpipe driver: the fragment
# shader reads a UBO from descriptor set 0 and another from set 7, the last
# set Vulkan's advertised maxBoundDescriptorSets allows. Set 7 lands one
# constant-buffer index past the end of the driver's descriptor table.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render + verify via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := ubo_lastset

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

# STRICT is 0 because refusing a shader that reaches past the descriptor table,
# and letting llvmpipe render it, is a correct outcome; the assertion is the
# centre pixel, not where it was produced. Set before common.mk.
STRICT := 0

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/ubo_lastset.vert $(SRC_DIR)/ubo_lastset.frag

include ../common.mk
