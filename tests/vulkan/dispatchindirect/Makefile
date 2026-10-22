# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# vkCmdDispatchIndirect with workgroup counts written by a shader in the same
# command buffer, with zero counts, an offset record, a large grid and
# host-written counts, and dispatches under conditional rendering.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run each case and check every workgroup and the grid
#
# STRICT is 1: the device must run every dispatch itself. Set before common.mk.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dispatchindirect

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

STRICT := 1
# Every count and predicate is read by the command processor, and every
# dispatch runs on the device.
EXPECT_NO_HOSTREAD := 1
EXPECT_ATTRIB := launches=25/25_draws=0/0

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/args.comp $(SRC_DIR)/mark.comp

include ../common.mk
