# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Subgroup quad operations and helper invocations in the fragment stage.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render the scissored triangle + check every pixel

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := quadops

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/quadops.vert $(SRC_DIR)/quadops.frag

# Subgroup operations need SPIR-V 1.3, the Vulkan 1.1 target environment.
GLSLC := glslc --target-env=vulkan1.1

include ../common.mk
