# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# 4:2:2 packed textures (G8B8G8R8_422, B8G8R8G8_422) sampled through a sampler
# Y'CbCr conversion from compute: every texel's luma and its pair's chroma.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run + check every texel of both formats

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_422

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/t422.comp

include ../common.mk
