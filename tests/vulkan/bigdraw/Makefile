# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# bigdraw -- draws with more primitives than one rasterizer pass can number:
# 70,000 triangles from the vertex stage, and 72,000 from a geometry stage with
# static and with dynamic counts, each image compared pixel for pixel.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run every case (OPTS="-c vs|gs|gsdyn" for one)

# SimX runs it functionally: the case is about which primitives reach the
# target and in what order, which timing does not change. WIDE because the
# vertex, geometry and front-end stages run every primitive in SIMT lanes, and
# 32x32 runs them in an eighth of the warp instructions.
CONFIGS += -DSIMX_FUNCTIONAL

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := bigdraw

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/big.vert $(SRC_DIR)/big.frag $(SRC_DIR)/big.geom

WIDE := 1

include ../common.mk
