# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Point and line rasterization test for the vortexpipe driver: points (and
# gl_PointCoord), rectangular, wide, Bresenham and smooth lines, stipple,
# polygon modes and instancing, each checked against the coverage Vulkan
# defines.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the case via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := primitives

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/primitives.vert \
           $(SRC_DIR)/primitives.frag

LDLIBS += -lm

include ../common.mk
