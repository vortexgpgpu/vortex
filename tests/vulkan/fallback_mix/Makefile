# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Device / llvmpipe / device interleave against one storage buffer. Covers the
# driver's invalidation of its device copy when work runs on the CPU fallback,
# which writes host memory with no gallium transfer to interpose on.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the three dispatches + check the result is i+3

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := fallback_mix

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

# The middle dispatch MUST really execute on llvmpipe: strict mode turns the
# fallback into a no-op, and a no-op cannot write the host buffer behind the
# driver's back, which is the exact situation under test. Set before common.mk.
STRICT := 0

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/add1.comp $(SRC_DIR)/add1_shared.comp

include ../common.mk
