# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Logic-op test for the vortexpipe driver: XOR against a known destination,
# whose result is neither operand.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the case via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := logicop

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/logicop.vert \
           $(SRC_DIR)/logicop_xor.frag

include ../common.mk
