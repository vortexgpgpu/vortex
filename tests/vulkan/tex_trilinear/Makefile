# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Mip-linear (trilinear) filtering for the vortexpipe driver: a two-level chain
# with flat, distinct levels sampled at explicit LOD 0, 0.5 and 1.
#
# Runs on the software sampler today -- any mipmapped sampler is routed there,
# and nothing can pin one to the fixed-function unit. The test fixes the answer
# so that it still has to hold once the routing moves.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the case via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_trilinear

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/tex_trilinear.vert \
           $(SRC_DIR)/tex_trilinear.frag

include ../common.mk
