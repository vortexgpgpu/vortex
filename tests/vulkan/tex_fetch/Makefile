# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# texelFetch / textureSize Vulkan test for the vortexpipe driver: each
# case uploads a small texture, fetches texels at integer coordinates in
# the fragment shader, verifies them in-shader and writes green/red.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run all cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_fetch

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/tex_fetch.vert \
           $(SRC_DIR)/fetch_rgba8.frag \
           $(SRC_DIR)/fetch_f32.frag \
           $(SRC_DIR)/fetch_i8.frag \
           $(SRC_DIR)/fetch_array.frag

include ../common.mk
