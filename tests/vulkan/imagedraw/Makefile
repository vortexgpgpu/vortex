# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# A storage image written from the fragment stage. The draw path relocates only
# buffer descriptors, so an image descriptor reaches the device still holding a
# host pointer.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- draw, then check the storage image and the attachment
#
# STRICT stays at the suite default of 1: llvmpipe stores to a storage image
# correctly, so a draw that fell back would pass while testing nothing.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk


PROJECT := imagedraw

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/imagedraw.vert \
           $(SRC_DIR)/imagedraw.frag \
           $(SRC_DIR)/imagedraw_vswrite.vert \
           $(SRC_DIR)/imagedraw_plain.frag

include ../common.mk
