# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Two colour attachments of DIFFERENT formats in one draw. Each render target
# carries its own merger state, so the format and the texel width are the
# attachment's own; only the channel order is shared across the set.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render both attachments + check each one's stored bytes
#
# STRICT stays at the suite default of 1 on purpose: llvmpipe encodes every
# combination correctly, so a draw that fell back would pass without testing
# anything.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := mrtfmt

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/mrtfmt.vert $(SRC_DIR)/mrtfmt.frag

include ../common.mk
