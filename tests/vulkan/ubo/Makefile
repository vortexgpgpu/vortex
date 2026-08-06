# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# UBO + push-constant fragment-shader descriptor test (W7) for the
# vortexpipe driver: both stages run on the Vortex device; the fragment
# stage reads its colour from a uniform buffer scaled by a push constant.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render + verify via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := ubo

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

LDLIBS += -lm

SHADERS := $(SRC_DIR)/ubo.vert $(SRC_DIR)/ubo.frag

include ../common.mk
