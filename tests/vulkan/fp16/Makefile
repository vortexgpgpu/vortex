# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# fp16 coverage test for the vortexpipe driver: fp16.comp runs
# binary16 arithmetic and converts on the Vortex device, checked
# exactly on the host. Skips honestly when shaderFloat16 is absent.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run on Vortex via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := fp16

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/fp16.comp

# float16 needs the Vulkan 1.2 SPIR-V environment for the Float16 capability.
GLSLC := glslc --target-env=vulkan1.2

include ../common.mk
