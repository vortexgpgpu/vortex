# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Storage-image coverage: imageLoad, imageStore and imageAtomicAdd on an
# R32_UINT 2D image, plus a re-seed through a buffer-to-image copy between
# dispatches.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run both rounds + check every texel

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := image

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/img_rw.comp $(SRC_DIR)/img_atomic.comp

include ../common.mk
