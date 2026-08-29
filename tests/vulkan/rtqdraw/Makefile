# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# A ray query in a fragment shader. The same scene as tests/vulkan/rtquery,
# reached through a draw instead of a dispatch: the two paths relocate
# descriptors through separate loops, and only the dispatch loop handles an
# acceleration structure.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- draw, then check the ray answers in the attachment
#
# NOTE: requires an RTU-enabled mesa-vortex install (pass MESA_PATH=<rtu
# prefix>); the RTU simx is selected by the EXT_RTU_ENABLE config below.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Build the simx runtime with the RTU extension so the device advertises
# VX_ISA_EXT_RTU; without it vortexpipe sees has_rtu=0 and lavapipe expands
# the ray query to a software BVH walk that falls back to llvmpipe.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)

PROJECT := rtqdraw

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/rtqdraw.vert \
           $(SRC_DIR)/rtqdraw.frag

# rayQueryEXT needs SPIR-V 1.4 -- compile against the Vulkan 1.2 env.
GLSLC := glslc --target-env=vulkan1.2

# Matches the other three ray-query tests: lavapipe's internal acceleration-
# structure build shaders (subgroup/atomic ops) run on llvmpipe, so a global
# refuse-fallback would trip on those builders rather than on anything under
# test. The device check in common.mk still holds the draw to vortexpipe.
STRICT := 0

include ../common.mk
