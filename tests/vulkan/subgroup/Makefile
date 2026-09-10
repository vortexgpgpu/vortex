# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Subgroup coverage test for the vortexpipe driver. subgroup.comp runs the
# ballot / vote / shuffle / scan family on the Vortex device and
# subgroup_ext.comp adds the relative shuffles and a clustered reduce; the
# host reproduces every cross-lane result.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run on Vortex via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := subgroup

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

# Order matters: main.c takes the core module as argv[1] and the
# relative-shuffle/clustered module as argv[2].
SHADERS := $(SRC_DIR)/subgroup.comp $(SRC_DIR)/subgroup_ext.comp

# GL_KHR_shader_subgroup_* lower to SPIR-V 1.3 group operations, which glslc
# only emits for Vulkan 1.1 or later; its default target is Vulkan 1.0.
GLSLC := glslc --target-env=vulkan1.1

include ../common.mk
