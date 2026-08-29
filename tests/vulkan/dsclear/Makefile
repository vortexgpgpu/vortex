# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Depth/stencil clear-value test for the vortexpipe driver: both planes are
# cleared to values no fill heuristic produces, and two quads either side of the
# depth clear report which values the device actually saw.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the case via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dsclear

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/dsclear.vert \
           $(SRC_DIR)/dsclear_near.frag \
           $(SRC_DIR)/dsclear_far.frag

include ../common.mk
