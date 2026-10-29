# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Robust texel fetches: texelFetch inside and outside a 2D array, a 3D image
# and a texel buffer, and through a null texel buffer, on a device with
# robustImageAccess2 and nullDescriptor.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run + check every probe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_robust

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/trobust.comp $(SRC_DIR)/tbnull.comp $(SRC_DIR)/pairs.comp

include ../common.mk
