# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# textureGather Vulkan test for the vortexpipe driver: each case gathers
# the 2x2 footprint around a coordinate, verifies the four texels and
# their GL gather order in-shader, and writes green/red.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run all cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_gather

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/tex_gather.vert \
           $(SRC_DIR)/gather_comp.frag \
           $(SRC_DIR)/gather_wrap.frag \
           $(SRC_DIR)/gather_array.frag

include ../common.mk
