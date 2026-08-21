# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Private-array compute test for the vortexpipe driver:
# kernel writing and reading a local array at runtime-computed indices.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the kernel via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := manydesc

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/manydesc.comp

include ../common.mk
