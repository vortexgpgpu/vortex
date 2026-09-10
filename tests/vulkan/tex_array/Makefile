# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# 2D-array texture test for the vortexpipe driver: layer selection,
# including a coordinate past the last layer, which Vulkan clamps.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run all cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_array

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/tex_array.vert \
           $(SRC_DIR)/array_layer0.frag \
           $(SRC_DIR)/array_layer2.frag \
           $(SRC_DIR)/array_clamp.frag

include ../common.mk
