# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# A sparse storage image two 64 KiB tiles wide, which the device's linear
# addressing cannot reach: a compute dispatch and a rasterizer-discard draw
# whose vertex shader stores into it must both be refused with the
# sparse-image reason and run on llvmpipe, and their results must be right.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run both + check every texel and the work placement

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := sparsefb

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

# The refused work has to run, on llvmpipe, for its results to be checked;
# STRICT would turn it into a no-op. Set before common.mk.
STRICT := 0
# Neither the dispatch nor the draw may run on the device, and both must be
# refused for the sparse image.
EXPECT_ATTRIB := launches=0/1_draws=0/1
EXPECT_CPU := sparse-image=2

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/sf_store.comp $(SRC_DIR)/sf_store.vert

include ../common.mk
