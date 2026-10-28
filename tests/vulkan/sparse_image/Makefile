# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Storage access to partially resident sparse images two or more 64 KiB tiles
# wide: 2D, 3D and 2D array views, texels of 1, 4 and 8 bytes, packed and
# per-channel encodes and a format-less view. Stores, sparse loads, plain loads
# and atomics must address each texel in its own tile and report its residency,
# and a rebind to new memory must be seen by the next dispatch.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run every case + check every texel and its tile memory

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := sparse_image

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/sp_r32_2d.comp $(SRC_DIR)/sp_r32_3d.comp $(SRC_DIR)/sp_r32_arr.comp \
           $(SRC_DIR)/sp_rgba8_2d.comp $(SRC_DIR)/sp_rgba8_arr.comp $(SRC_DIR)/sp_rg16_3d.comp \
           $(SRC_DIR)/sp_r8_2d.comp $(SRC_DIR)/sp_rgba16_arr.comp $(SRC_DIR)/sp_noformat_2d.comp

include ../common.mk
