# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# A ray query left open across a workgroup barrier, in a workgroup of two
# warps. A trace parked at a candidate holds the core's RTU slot, so the other
# warp's trace could not start and that warp would never reach the barrier the
# first one waits at: vortexpipe refuses the shader for the device
# ("trace-barrier") and it runs on llvmpipe, where every value is checked
# against the host's traversal (rtquery_full built with RTQ_BARRIER).
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run under vortexpipe and compare

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)
CONFIGS := $(if $(findstring VX_CFG_RTU_BVH_WIDTH,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_RTU_BVH_WIDTH=4)

WIDE := 1
STRICT := 0
EXPECT_CPU := trace-barrier=1

PROJECT := rtqbarrier

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/rtquery_full

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/rtquery_full.comp

GLSLC := glslc --target-env=vulkan1.2 -DRTQ_BARRIER
CFLAGS += -DRTQ_BARRIER

LDLIBS += -lm

include ../common.mk
