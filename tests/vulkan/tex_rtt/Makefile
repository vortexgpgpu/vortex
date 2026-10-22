# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Write a texture, sample it through the TEX unit, write it again and sample
# it again: by a render pass, then by a compute shader's image stores, in turn,
# with a different pattern every time.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the rounds on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_rtt

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/rtt.vert $(SRC_DIR)/rtt_write.frag $(SRC_DIR)/rtt_write.comp \
           $(SRC_DIR)/rtt_sample.frag

include ../common.mk

# The writers #include the pattern the host checks against.
rtt_write.frag.spv rtt_write.comp.spv: $(SRC_DIR)/rtt_pattern.glsl
