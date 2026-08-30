# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Multi-descriptor-set compute test for the vortexpipe driver:
# kernel reaching its storage buffers through two bound descriptor sets.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the kernel via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := csmultiset

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

# STRICT is 0 because refusing a dispatch whose descriptors the launch
# relocation cannot place, and letting llvmpipe run it, is a correct outcome;
# the assertion is the contents of the buffers, not where they were written.
# Set before common.mk.
STRICT := 0

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/csmultiset.comp

include ../common.mk
