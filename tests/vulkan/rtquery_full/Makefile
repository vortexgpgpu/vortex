# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Every VK_KHR_ray_query value, per invocation, against the host's own
# traversal of the scene: committed and candidate records, confirm, ignore,
# terminate, generated procedural hits, ray flags and cull masks, two queries
# of an array, a query left open and re-traced, and a null acceleration
# structure.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- trace on the RTU via vortexpipe and compare

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The device advertises VX_ISA_EXT_RTU only when the runtime is built with it;
# without it the ray queries lower to lavapipe's software walk.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)
CONFIGS := $(if $(findstring VX_CFG_RTU_BVH_WIDTH,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_RTU_BVH_WIDTH=4)

# The acceleration-structure build kernels need a 1024-wide workgroup to run on
# the device, so the test runs on the 32x32 core and STRICT holds every launch
# -- the builds and the query -- to the device.
WIDE := 1
STRICT := 1

PROJECT := rtquery_full

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/rtquery_full.comp

# rayQueryEXT needs SPIR-V 1.4 -- compile against the Vulkan 1.2 env.
GLSLC := glslc --target-env=vulkan1.2

LDLIBS += -lm

include ../common.mk
