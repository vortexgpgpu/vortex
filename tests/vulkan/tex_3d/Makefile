# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# 3D (volume) texture test for the vortexpipe driver: depth-slice selection
# and an inter-slice blend.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run all cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_3d

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/tex_3d.vert \
           $(SRC_DIR)/vol_slice0.frag \
           $(SRC_DIR)/vol_slice3.frag

include ../common.mk
