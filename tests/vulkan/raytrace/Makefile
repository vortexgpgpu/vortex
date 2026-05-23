# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Ray-query smoke test for the vortexpipe driver (Phase 7, step 1):
# casts primary rays at a one-triangle acceleration structure. The
# ray tracing runs on lavapipe's software path -- the correctness
# oracle later Vortex-SIMT RT phases are validated against.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- trace the rays via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := raytrace

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/raytrace.comp

# rayQueryEXT needs SPIR-V 1.4 -- compile against the Vulkan 1.2 env.
GLSLC := glslc --target-env=vulkan1.2

include ../common.mk
