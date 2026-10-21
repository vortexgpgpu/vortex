# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Float-atomic test for the vortexpipe driver: atomic_float.comp contends
# add/min/max/exchange on a storage buffer, shared memory, a storage image and
# a buffer device address, and the host checks what any serialisation of the
# atomics must produce, exactly.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run on Vortex via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := atomic_float

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/atomic_float.comp

GLSLC := glslc --target-env=vulkan1.2

STRICT := 1

include ../common.mk
