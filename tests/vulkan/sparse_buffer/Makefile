# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Partially resident sparse buffers in compute: storage reads and writes,
# another buffer aliased on the same memory, a uniform buffer, the indirect
# dispatch arguments and a device address, all on sparse pages, across a
# rebind with no other call.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run both passes + check every word

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := sparse_buffer

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/sb.comp

include ../common.mk
