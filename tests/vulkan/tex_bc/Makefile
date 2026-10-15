# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Block-compressed textures (BC1-BC7) sampled from their compressed blocks.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- fetch every texel of each format + check its decode

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_bc

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/tex_bc.vert $(SRC_DIR)/tex_bc.frag

LDLIBS += -lm

include ../common.mk
