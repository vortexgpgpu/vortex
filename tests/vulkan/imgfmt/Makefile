# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Storage images written without a format: BGRA8, A8 and the float, snorm and
# packed formats, each encoded from the view at run time.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- store into each format + check every field

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := imgfmt

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/imgfmt.comp

LDLIBS += -lm

include ../common.mk
