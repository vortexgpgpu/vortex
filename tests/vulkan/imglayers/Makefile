# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Storage images with layers: 2D arrays, 3D and cube images, addressed per
# layer, slice and face for load, store and atomics.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- fill each image + check every layer

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := imglayers

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/imglayers.comp

include ../common.mk
