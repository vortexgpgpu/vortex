# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# 1D texture test for the vortexpipe driver.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the case via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_1d

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/tex_1d.vert \
           $(SRC_DIR)/tex1d_sample.frag

include ../common.mk
