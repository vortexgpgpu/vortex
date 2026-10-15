# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# 4x multisample resolves for the vortexpipe driver: render-pass resolve and
# vkCmdResolveImage of RGBA8 (average), R32F (average) and RGBA8UI (sample 0),
# a sub-rectangle resolve and a resolve of one layer of a layered image, each
# from sources holding a distinct value in every sample.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := msaa_resolve

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/resolve.vert \
           $(SRC_DIR)/fill_unorm.frag \
           $(SRC_DIR)/fill_float.frag \
           $(SRC_DIR)/fill_uint.frag

LDLIBS += -lm

include ../common.mk
