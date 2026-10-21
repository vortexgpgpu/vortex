# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# one draw with more primitives than a RASTER pass addresses
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- draw 33792 near-clipped triangles (67584 records) in one
#                      instanced vkCmdDraw, and two stippled line strips longer
#                      than a pass; check every pixel
#
# The RASTER unit numbers a pass's primitives with a VX_RASTER_PID_BITS pid, so
# the driver splits a larger draw into passes. STRICT stays at the suite default
# of 1: llvmpipe has no such limit, so a fallback would pass untested. WIDE
# because the front end sets up every triangle in SIMT lanes, and 32x32 runs
# them in an eighth of the warp instructions.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := manyprims

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/manyprims.vert $(SRC_DIR)/manyprims.frag \
           $(SRC_DIR)/manyprims_line.vert $(SRC_DIR)/manyprims_line.frag

WIDE := 1

include ../common.mk
