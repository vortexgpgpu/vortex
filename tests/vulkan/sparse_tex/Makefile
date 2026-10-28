# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Sparse fetches, samples and gathers of partially resident sparse textures --
# a 2D array with an unbound level and a 3D texture, tiles in a checkerboard --
# from compute and from a fragment shader. Residency codes must follow the
# footprint, values the tiles, and both a rebind with no other call.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run + check every texel and pixel

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := sparse_tex

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/st_arr.comp $(SRC_DIR)/st_3d.comp $(SRC_DIR)/st_draw.vert \
           $(SRC_DIR)/st_draw.frag

include ../common.mk
