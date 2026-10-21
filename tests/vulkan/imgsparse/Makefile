# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# sparseImageLoadARB on ordinary storage images: 2D, 2D array, cube and 3D
# R32_UINT views, an RGBA8 view and a format-less read. Every texel must come
# back with its seeded value and a resident residency code. Then imageLoad and
# imageStore through a view of one layer of a fully bound sparse 2D array.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run every case + check every texel

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := imgsparse

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/sl_2d.comp $(SRC_DIR)/sl_2darray.comp $(SRC_DIR)/sl_cube.comp \
           $(SRC_DIR)/sl_3d.comp $(SRC_DIR)/sl_rgba8.comp $(SRC_DIR)/sl_noformat.comp \
           $(SRC_DIR)/sl_layer.comp

include ../common.mk
