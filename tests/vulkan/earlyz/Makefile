# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Depth-test invariants an early-Z stage must preserve: draw order must not
# change the image, and a shader-written depth must beat its own plane.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := earlyz

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/earlyz.vert \
           $(SRC_DIR)/earlyz.frag \
           $(SRC_DIR)/earlyz_depth.frag

include ../common.mk
