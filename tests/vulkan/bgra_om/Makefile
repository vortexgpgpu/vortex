# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Output-merger state that is expressed in channel order, rendered to a
# B8G8R8A8_UNORM attachment: the colour write mask and the constant blend
# colour. tests/vulkan/bgra covers the fragment colour itself; neither of these
# two is visible there, because a full write mask and a disabled blend are the
# only channel-order state a plain draw touches.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render both phases + check the memory byte order
#
# STRICT stays at the suite default of 1 on purpose: llvmpipe gets the channel
# order right, so a draw that fell back would pass without testing anything.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := bgra_om

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/bgra_om.vert $(SRC_DIR)/bgra_om.frag

include ../common.mk
