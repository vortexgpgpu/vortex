# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# A draw inside a conditional-rendering block. llvmpipe tests the predicate at
# the top of its draw, so a draw that bypasses it renders whatever the
# predicate said to discard.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render under each predicate + check the pixels
#
# STRICT is 0 because refusing the draw and rendering it on llvmpipe is a
# correct outcome; the assertion is the pixels, not where they were produced.
# Set before common.mk.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := condrender

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

STRICT := 0

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/condrender.vert $(SRC_DIR)/condrender.frag

include ../common.mk
