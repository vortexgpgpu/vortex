# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# 4x depth/stencil render-pass resolves for the vortexpipe driver: D32F depth
# SAMPLE_ZERO and AVERAGE, a 24- or 16-bit depth AVERAGE, and a combined
# depth/stencil SAMPLE_ZERO over the whole target and over a partial render
# area, from a sloped plane whose four samples hold four different depths and
# a stencil pattern that differs between sample 0 and the others.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := msaa_dsresolve

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/plane.vert \
           $(SRC_DIR)/plane.frag \
           $(SRC_DIR)/plane_even.frag

STRICT := 1

LDLIBS += -lm

FF_RASTER := 1
FF_OM := 1

include ../common.mk
