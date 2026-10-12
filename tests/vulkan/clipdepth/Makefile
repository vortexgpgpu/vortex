# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Clipping and depth-range test for the vortexpipe driver: user clip and cull
# distances, depth clipping against the viewport's depth range, depth clamp,
# and depth bias, each checked against the exact pixel mask it must produce.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the case via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := clipdepth

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/clipdepth.vert \
           $(SRC_DIR)/clipdepth.frag

include ../common.mk
