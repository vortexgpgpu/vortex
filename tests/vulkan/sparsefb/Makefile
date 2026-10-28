# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Stores into a sparse storage image two 64 KiB tiles wide, which linear
# addressing does not reach: a compute dispatch and a rasterizer-discard draw
# whose vertex shader stores into it must both run on the device, and their
# results must be right.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run both + check every texel and the work placement

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := sparsefb

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

# Both the dispatch and the draw run on the device.
EXPECT_ATTRIB := launches=1/1_draws=1/1

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/sf_store.comp $(SRC_DIR)/sf_store.vert

include ../common.mk
