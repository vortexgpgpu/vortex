# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Integer colour attachments of at most 32 bits, stored by the fixed-function
# output merger: saturation, byte order, writemasks and the logic op.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render each phase + check the stored bytes
#
# FF_OM: the software merger stores the same bytes, so the run fails if any
# draw merged there.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := intrt

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/intrt.vert $(SRC_DIR)/intrt_u.frag $(SRC_DIR)/intrt_i.frag

FF_OM := 1

include ../common.mk
