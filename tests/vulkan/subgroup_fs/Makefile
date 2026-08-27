# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Subgroup operations in the FRAGMENT stage: gl_SubgroupSize must be the device
# warp width, not the width the host rasterizer finalized the shader at.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- shade, then compare against the advertised subgroup size
#
# STRICT stays at the suite default of 1: llvmpipe reports its own width
# correctly, so a draw that fell back would pass while testing nothing.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Subgroup intrinsics need SPIR-V 1.3, which glslc emits only for a Vulkan 1.1
# target env; the suite default targets 1.0.
GLSLC := glslc --target-env=vulkan1.1

PROJECT := subgroup_fs

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/subgroup_fs.vert $(SRC_DIR)/subgroup_fs.frag

include ../common.mk
