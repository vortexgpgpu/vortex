# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Blending INTO an sRGB colour attachment -- the decode side of the transfer
# function, which is only reachable through a blend and which every other sRGB
# test leaves unexercised.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- draw a destination, blend over it, check the stored bytes
#
# STRICT stays at the suite default of 1 on purpose: llvmpipe blends these
# correctly, so a draw that fell back would pass without testing anything.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := srgbblend

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/srgbblend.vert $(SRC_DIR)/srgbblend_base.frag \
           $(SRC_DIR)/srgbblend_src.frag

include ../common.mk
