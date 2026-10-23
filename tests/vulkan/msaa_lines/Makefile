# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# 4x lines rasterized at the pixel centre (Bresenham and smooth) on the
# fixed-function rasterizer and merger: the rasterizer runs at one sample and a
# covered pixel takes every sample. The primitives test, built for 4x.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := msaa_lines

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/primitives

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/primitives.vert \
           $(SRC_DIR)/primitives.frag

CFLAGS += -DPRIM_SAMPLES=4
LDLIBS += -lm

FF_RASTER := 1
FF_OM := 1

include ../common.mk
