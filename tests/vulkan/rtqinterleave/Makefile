# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Two ray queries open at once and proceeded in turn. The RTU stages one
# returned candidate per warp, so vortexpipe refuses the shader for the device
# ("nested-trace") and it runs on llvmpipe, where every value of both queries
# is checked against the host's traversal (rtquery_full built with
# RTQ_INTERLEAVE).
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run under vortexpipe and compare

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)
CONFIGS := $(if $(findstring VX_CFG_RTU_BVH_WIDTH,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_RTU_BVH_WIDTH=4)

WIDE := 1
STRICT := 0
EXPECT_CPU := nested-trace=1

PROJECT := rtqinterleave

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/rtquery_full

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/rtquery_full.comp

GLSLC := glslc --target-env=vulkan1.2 -DRTQ_INTERLEAVE
CFLAGS += -DRTQ_INTERLEAVE

LDLIBS += -lm

include ../common.mk
