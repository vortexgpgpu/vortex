# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Descriptor-count compute test for the vortexpipe driver:
# kernel writing more distinct storage buffers than the descriptor scan records.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the kernel via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := manydesc

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

# STRICT is 0 because refusing a shader whose descriptors the relocation cannot
# carry, and letting llvmpipe run it, is a correct outcome; the assertion is the
# contents of the buffers, not where they were written. Set before common.mk.
STRICT := 0

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/manydesc.comp

include ../common.mk
