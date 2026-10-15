# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# 4x multisample interpolation functions and sample built-ins for the
# vortexpipe driver: interpolateAtOffset, interpolateAtSample and
# interpolateAtCentroid of a window-position input, gl_SampleMaskIn at sample
# and at pixel rate, and textureSamples of a sampler2DMS, each written by the
# fragment shader into a storage buffer and checked on the host.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := msaa_interp

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/interp.vert \
           $(SRC_DIR)/interp_offset.frag \
           $(SRC_DIR)/interp_sample.frag \
           $(SRC_DIR)/interp_centroid.frag \
           $(SRC_DIR)/mask_sample.frag \
           $(SRC_DIR)/mask_pixel.frag \
           $(SRC_DIR)/tex_samples.frag

STRICT := 1

LDLIBS += -lm

include ../common.mk
