# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Channel order of a B8G8R8A8_UNORM colour attachment, which the driver
# advertises alongside R8G8B8A8_UNORM while every other test uses the latter.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render a known colour + check the memory byte order
#
# STRICT stays at the suite default of 1 on purpose: llvmpipe gets the channel
# order right, so a draw that fell back would pass without testing anything.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := bgra

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/bgra.vert $(SRC_DIR)/bgra.frag

include ../common.mk
