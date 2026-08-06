# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Multi-set UBO descriptor test for the vortexpipe driver (issue I5):
# the fragment shader reads a UBO from descriptor set 0 AND another from
# descriptor set 1, combining both. A correct render proves the FS
# descriptor relocation handles sets >= 1 (not just set 0) on the Vortex
# device.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render + verify via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := ubo_multiset

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/ubo_multiset.vert $(SRC_DIR)/ubo_multiset.frag

include ../common.mk
