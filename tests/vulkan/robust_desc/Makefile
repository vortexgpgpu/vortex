# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Descriptor robustness the pipeline asks for, and none it does not:
#   - robustBufferAccess without robustImageAccess: storage texel buffer stores
#     past the view's end must not reach the buffer bound after it;
#   - nullDescriptor: a storage image slot rewritten to VK_NULL_HANDLE over a
#     live image sizes as 0, reads as 0 and leaves the old image untouched.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run + check both

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := robust_desc

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/texbuf_store.comp $(SRC_DIR)/null_image.comp

include ../common.mk
