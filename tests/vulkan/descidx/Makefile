# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Storage-buffer, uniform-texel-buffer, storage-image and format-less
# storage-image descriptor arrays indexed per lane, in a compute and a fragment
# shader: every lane reaches the descriptor its index selects, so the driver can
# relocate none ahead of the work and the device translates each pointer. The
# buffers share one allocation, and the draw's vertex shader stores through a
# fixed descriptor beside the indexed fragment stores.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run on Vortex via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := descidx

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/descidx.comp $(SRC_DIR)/descidx.vert $(SRC_DIR)/descidx.frag

GLSLC := glslc --target-env=vulkan1.2

include ../common.mk
