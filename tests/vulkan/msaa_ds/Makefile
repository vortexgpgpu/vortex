# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# 4x multisample per-sample depth testing on one RGBA8 attachment, with D32F and
# then D24S8 depth: a sloped triangle crossing a flat quad, where which one wins
# is decided per sample by the depth at that sample. The draws stay on the
# fixed-function rasterizer and merger.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := msaa_ds

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/mrt.vert \
           $(SRC_DIR)/ds.frag \
           $(SRC_DIR)/mrt_quad.vert \
           $(SRC_DIR)/mrt_read.frag

FF_RASTER := 1
FF_OM := 1

include ../common.mk
