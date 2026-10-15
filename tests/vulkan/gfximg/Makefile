# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# A fragment-stage storage image read framed by buffer atomics, the CTS's
# descriptor-check pattern: the one fragment of a 1x1 pass must read the texel
# the host wrote and see both of its atomics land.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run on Vortex via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := gfximg

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/gfximg.vert $(SRC_DIR)/gfximg.frag

include ../common.mk
