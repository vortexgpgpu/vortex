# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Sixteen varying scalars against a front end carrying twelve interpolation
# planes. The fragment shader returns the varying that does not fit, so a
# driver that silently drops the remainder shows up as a wrong colour.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render the triangle + check the centre pixel
#
# STRICT is 0 because refusing the shader and rendering it on llvmpipe is a
# correct outcome; the assertion is the colour, not where it was produced.
# Set before common.mk.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := manyvary

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

STRICT := 0

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/manyvary.vert $(SRC_DIR)/manyvary.frag

include ../common.mk
