# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Depth-test Vulkan test for the vortexpipe driver: two overlapping
# triangles at different depths, rendered with the Vortex OM unit's
# depth test on D32F, D16 and X8_D24, plus D16 precision ties.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render via vortexpipe (RASTER + OM)

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := depth

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/depth.vert $(SRC_DIR)/depth.frag \
           $(SRC_DIR)/depth_frag.frag $(SRC_DIR)/depth_tie.vert

include ../common.mk
