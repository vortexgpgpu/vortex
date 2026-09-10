# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# An occlusion query around a draw the device runs. The query is counted by the
# rasterizer, so a draw that bypasses llvmpipe's leaves the counter at zero.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render the triangle + check the query result
#
# STRICT is 0 because refusing the draw and rendering it on llvmpipe is a
# correct outcome; the assertion is the count, not where it was produced.
# Set before common.mk.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := occlusion

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

STRICT := 0

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/occlusion.vert $(SRC_DIR)/occlusion.frag

include ../common.mk
