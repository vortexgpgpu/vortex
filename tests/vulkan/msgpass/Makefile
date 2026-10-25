# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Vulkan memory-model message passing: device scope with an image or buffer
# payload (also from a vertex shader), and subgroup scope through shared memory
# with a partial subgroup.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run on Vortex via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := msgpass

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/mp_image.comp $(SRC_DIR)/mp_buffer.comp $(SRC_DIR)/mp_subgroup.comp \
           $(SRC_DIR)/mp_vertex.vert

GLSLC := glslc --target-env=vulkan1.2

# The CTS runtime shape: device-scope partners run on different cores, whose
# L1s only a device fence keeps from serving each other stale lines.
WIDE := 1
CONFIGS += -DVX_CFG_NUM_CORES=4 -DVX_CFG_L2_ENABLE

include ../common.mk
