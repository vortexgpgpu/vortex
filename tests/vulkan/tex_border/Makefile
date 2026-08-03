# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# CLAMP_TO_BORDER test for the vortexpipe driver: coordinates outside
# [0,1] must return the sampler's border colour, not the edge texel.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the case via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_border

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/tex_border.vert \
           $(SRC_DIR)/border_clamp.frag

include ../common.mk
