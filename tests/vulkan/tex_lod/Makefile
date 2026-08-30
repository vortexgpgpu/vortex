# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Mip-level / LOD test for the vortexpipe driver: explicit textureLod on a
# mip chain whose levels are distinct colours, and a mip-linear sample that
# must land between two of them.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run all cases via vortexpipe on SimX
#   make run-rtlsim -- the same on the RTL model

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_lod

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/tex_lod.vert \
           $(SRC_DIR)/lod_base.frag \
           $(SRC_DIR)/lod_level2.frag \
           $(SRC_DIR)/lod_trilinear.frag

include ../common.mk
