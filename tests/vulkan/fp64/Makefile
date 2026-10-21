# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# fp64 test for the vortexpipe driver: double arithmetic, conversions,
# comparisons, rounding, subgroup ops and flat double varyings, through the
# toolchain's soft-float library on rv32 (no D extension) and natively on rv64,
# checked bit-exactly on the host.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run on Vortex via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := fp64

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/fp64.comp $(SRC_DIR)/fp64.frag $(SRC_DIR)/fp64.vert

GLSLC := glslc --target-env=vulkan1.2

STRICT := 1

LDLIBS += -lm

include ../common.mk
