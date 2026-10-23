# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# rtquery_full on vortexpipe's llvmpipe fallback: the query dispatch is handed
# to llvmpipe though the RTU could run it, so it must run lavapipe's software
# walk of the same scene, and every value is checked against the host's
# traversal as on the device. The acceleration-structure builds stay on the
# device (they are specialized for the warp's subgroup size).
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- trace on llvmpipe under vortexpipe and compare

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)
CONFIGS := $(if $(findstring VX_CFG_RTU_BVH_WIDTH,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_RTU_BVH_WIDTH=4)

WIDE := 1
STRICT := 0
FORCE_LLVMPIPE := launch
EXPECT_CPU := forced=1

PROJECT := rtquery_full_cpu

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/rtquery_full

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/rtquery_full.comp

GLSLC := glslc --target-env=vulkan1.2

LDLIBS += -lm

include ../common.mk
