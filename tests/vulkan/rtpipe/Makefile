# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# vkCmdTraceRaysKHR with raygen, miss, closest-hit, any-hit, intersection and
# callable shaders: any-hit alpha test and terminate, procedural hits with and
# without any-hit, both miss shaders, closest-hit shaders that trace again
# (recursion depth 3), a callable calling a callable, sbtRecordOffset/stride
# indexing and shader-record data, an opaque/non-opaque tie at one distance and
# per-lane ray flags within a warp. Every launch cell's record is compared
# against the host's own traversal.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- trace on the RTU via vortexpipe and compare

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The device advertises VX_ISA_EXT_RTU only when the runtime is built with it.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)
CONFIGS := $(if $(findstring VX_CFG_RTU_BVH_WIDTH,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_RTU_BVH_WIDTH=4)

# The acceleration-structure build kernels need a 1024-wide workgroup to run on
# the device, so the test runs on the 32x32 core and STRICT holds every launch
# -- the builds and the trace -- to the device.
WIDE := 1
STRICT := 1

PROJECT := rtpipe

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

# Order matters: main.c takes the stages in this order.
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
