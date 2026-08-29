# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# W8 draw-instancing Vulkan test for the vortexpipe driver: a
# 3-instance draw whose vertex stage runs on the Vortex device and
# resolves gl_InstanceIndex, rendering three separated triangles.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render the instanced draw via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := instanced

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/instanced.vert $(SRC_DIR)/instanced.frag

include ../common.mk
