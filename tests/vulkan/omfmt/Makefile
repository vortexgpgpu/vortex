# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Colour attachment formats the output merger encodes rather than stores
# verbatim: R8 and RG8, whose texels are narrower than a word, and the two sRGB
# orders, whose stored bytes are not the shader's values.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render each format + check the stored bytes
#
# STRICT stays at the suite default of 1 on purpose: llvmpipe encodes all four
# correctly, so a draw that fell back would pass without testing anything.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := omfmt

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/omfmt.vert $(SRC_DIR)/omfmt.frag

include ../common.mk
