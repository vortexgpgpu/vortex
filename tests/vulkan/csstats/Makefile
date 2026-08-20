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

PROJECT := csstats

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

# STRICT is 0 because refusing the dispatch and letting llvmpipe run and count
# it is a correct outcome; the assertion is the count, not where it was
# produced. Set before common.mk.
STRICT := 0

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/csstats.comp

include ../common.mk
