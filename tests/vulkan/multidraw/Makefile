# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Multi-draw depth-persistence test for the vortexpipe driver: the near
# and far triangles of the `depth` test issued as two separate draws in
# one renderpass, checking the depth attachment persists across draws.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render via vortexpipe (RASTER + OM)

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := multidraw

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/near.vert $(SRC_DIR)/far.vert $(SRC_DIR)/depth.frag

include ../common.mk
