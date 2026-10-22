# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Draws inside conditional-rendering blocks, normal and inverted, with the
# predicate written by the host or copied from an occlusion query. llvmpipe
# tests the predicate at the top of its draw, so a draw that bypasses it renders
# whatever the predicate said to discard.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render under each predicate + check the pixels
#
# STRICT is 1: the device tests the predicate and runs the draws itself. Set
# before common.mk.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := condrender

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

STRICT := 1
# The predicate words are read by the command processor, never by the host.
EXPECT_NO_HOSTREAD := 1

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/condrender.vert $(SRC_DIR)/condrender.frag

include ../common.mk
