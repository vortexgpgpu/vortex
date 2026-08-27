# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# gl_VertexIndex / firstVertex test for the vortexpipe driver: a vertex-less
# draw at a non-zero firstVertex must render the vertices that offset selects.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the case via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := firstvertex

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/firstvertex.vert \
           $(SRC_DIR)/firstvertex.frag

include ../common.mk
