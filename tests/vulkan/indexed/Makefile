# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Indexed-draw test for the vortexpipe driver: a quad rendered via
# vkCmdDrawIndexed (index buffer [0,1,2,0,2,3], shared vertices). Validates the
# Vortex on-device indexed path — gl_VertexIndex resolved through the index
# buffer, the front end assembling triangles from the per-index VS output.
#
#   make run-simx   -- render via vortexpipe (indexed draw on Vortex)

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := indexed

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/quad.vert $(SRC_DIR)/indexed.frag

include ../common.mk
