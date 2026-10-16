# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Pipeline-statistics queries around draws and a dispatch the device runs, each
# counter asserted exactly: instanced, off-viewport, rasterizer-discard,
# cull-all, two-draw and multi-draw graphics work, and a 2-row compute grid.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render + check the query results
#
# STRICT is 1: the draws must stay on the device while the query is open. Set
# before common.mk.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := gfxstats

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

STRICT := 1

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/gfxstats.vert $(SRC_DIR)/gfxstats.frag $(SRC_DIR)/gfxstats.comp

include ../common.mk
