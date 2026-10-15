# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Colour attachment formats described field by field and stored by the software
# merger: snorm, 16-bit, 2-10-10-10, 12-byte, packed 16-bit and alpha-only.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render each format + check every stored field

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := rtfield

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/rtfield.vert $(SRC_DIR)/rtfield_f.frag $(SRC_DIR)/rtfield_u.frag \
           $(SRC_DIR)/rtfield_i.frag

LDLIBS += -lm

include ../common.mk
