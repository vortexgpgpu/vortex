# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Two-sided stencil test for the vortexpipe driver: a first pass marks the
# stencil buffer, a second draws only where the mark admits it.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the case via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := stencil

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/stencil.vert \
           $(SRC_DIR)/stencil_pass.frag

include ../common.mk
