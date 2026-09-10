# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Repeated-draw test for the vortexpipe driver: one vertex buffer, four
# successive draws. Correctness is that all four land; the reason the test
# exists is that it is the only workload in the suite that draws twice from one
# vertex buffer, which is what a per-resource device allocation is for.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render four coincident draws + check the accumulated sum

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := redraw

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/redraw.vert $(SRC_DIR)/redraw.frag

include ../common.mk
