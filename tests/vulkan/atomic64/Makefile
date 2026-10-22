# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# 64-bit integer atomic test for the vortexpipe driver: atomic64.comp contends
# every 64-bit integer atomic on a storage buffer, shared memory and a buffer
# device address, and the host checks what any serialisation of the atomics
# must produce, exactly.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run on Vortex via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := atomic64

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/atomic64.comp

GLSLC := glslc --target-env=vulkan1.2

STRICT := 1

include ../common.mk
