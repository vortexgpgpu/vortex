# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Storage views of slices 14..17 of a sparse 3D image, which straddle its two
# tiles: a sliced 3D view, read on the device, and a 2D-array view, which
# Vulkan does not allow on a sparse image (VUID-VkImageCreateInfo-flags-09403)
# and the device does not address, so that dispatch runs on llvmpipe.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run both dispatches + check the views and placement

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := sparse_layers3d

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

STRICT := 0
# The 2D-array view's dispatch on llvmpipe, the sliced view's on the device.
EXPECT_ATTRIB := launches=1/2
EXPECT_CPU := sparse-image=1

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/sl_slice.comp $(SRC_DIR)/sl_load.comp

include ../common.mk
