# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Textured-quad Vulkan test for the vortexpipe driver: the vertex and
# fragment stages both run on the Vortex device, the fragment stage
# sampling the Vortex TEX hardware unit.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render the textured quad via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := textured

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/textured.vert $(SRC_DIR)/textured.frag

include ../common.mk
