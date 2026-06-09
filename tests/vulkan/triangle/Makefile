# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Hello-triangle Vulkan test for the vortexpipe driver: the vertex
# stage runs on the Vortex device, the fragment stage on llvmpipe.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render the triangle via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := triangle

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/triangle.vert $(SRC_DIR)/triangle.frag

include ../common.mk
