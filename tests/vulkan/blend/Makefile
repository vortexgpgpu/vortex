# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Blend test for the vortexpipe driver: a constant-colour factor and a
# separate alpha equation, neither of which any other state can produce.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the case via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := blend

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/blend.vert \
           $(SRC_DIR)/blend_const.frag

include ../common.mk
