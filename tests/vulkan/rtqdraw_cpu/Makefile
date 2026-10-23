# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# rtqdraw on vortexpipe's llvmpipe fallback: the draw whose fragment shader
# runs the ray query is handed to llvmpipe though the device could run it, so
# the fragment stage must walk the scene in software, and the hit count is
# checked as on the device. The acceleration-structure builds stay on the
# device (they are specialized for the warp's subgroup size).
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- draw on llvmpipe under vortexpipe and compare

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)

WIDE := 1
STRICT := 0
FORCE_LLVMPIPE := draw
EXPECT_CPU := forced=1

PROJECT := rtqdraw_cpu

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/rtqdraw

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/rtqdraw.vert \
           $(SRC_DIR)/rtqdraw.frag

GLSLC := glslc --target-env=vulkan1.2

include ../common.mk
