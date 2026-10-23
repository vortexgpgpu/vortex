# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# rtpipe on vortexpipe's llvmpipe fallback: the trace-rays launch is handed to
# llvmpipe though the RTU could run it, so it must run lavapipe's software
# ray-tracing pipeline, and every word of every launch cell is checked against
# the host's traversal as on the device. The acceleration-structure builds stay
# on the device (they are specialized for the warp's subgroup size).
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

PROJECT := rtpipe_cpu

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/rtpipe

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/rtpipe.rgen \
           $(SRC_DIR)/rtpipe_miss0.rmiss \
           $(SRC_DIR)/rtpipe_miss1.rmiss \
           $(SRC_DIR)/rtpipe_chit0.rchit \
           $(SRC_DIR)/rtpipe_chit1.rchit \
           $(SRC_DIR)/rtpipe_chit2.rchit \
           $(SRC_DIR)/rtpipe_ahit1.rahit \
           $(SRC_DIR)/rtpipe_ahit2.rahit \
           $(SRC_DIR)/rtpipe.rint \
           $(SRC_DIR)/rtpipe_call0.rcall \
           $(SRC_DIR)/rtpipe_call1.rcall

GLSLC := glslc --target-env=vulkan1.2

LDLIBS += -lm

include ../common.mk

$(SPVS): $(SRC_DIR)/rtpipe.glsl $(SRC_DIR)/rtpipe_hit.glsl
