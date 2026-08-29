# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# What a compute shader can ask about its own dispatch: gl_NumWorkGroups and an
# unsized storage array's .length(). Both come from the device at run time --
# one from a CSR, the other from the descriptor -- and neither can be folded to
# a constant at compile time.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- dispatch, then check both answers against what was asked
#                      for

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dispatchq

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/dispatchq.comp

include ../common.mk
