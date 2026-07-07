# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Triangle-strip test for the vortexpipe driver: a quad rendered as a
# non-indexed VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP (4 verts -> 2 triangles).
# Validates the W8 host-side topology translation — the strip is expanded to a
# triangle-LIST index array and run through the list-native device front end.
#
#   make run-simx   -- render via vortexpipe (strip translated on Vortex)

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tristrip

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/strip.vert $(SRC_DIR)/strip.frag

include ../common.mk
