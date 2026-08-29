# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Depth-compare (shadow) sampling test for the vortexpipe driver: each
# case samples a depth texture through a comparison sampler and checks
# the 0/1 result in-shader.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run all cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_shadow

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/tex_shadow.vert \
           $(SRC_DIR)/shadow_less.frag \
           $(SRC_DIR)/shadow_greater.frag \
           $(SRC_DIR)/shadow_array.frag

include ../common.mk
