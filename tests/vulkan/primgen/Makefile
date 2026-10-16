# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# VK_EXT_primitives_generated_query around draws the device runs: a triangle
# strip, a line list, the strip under rasterizer discard and the strip under a
# query on vertex stream 1, each in its own query.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render + check the query results
#
# STRICT is 1: the draws must stay on the device while the query is open. Set
# before common.mk.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := primgen

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

STRICT := 1

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/primgen.vert $(SRC_DIR)/primgen.frag

include ../common.mk
