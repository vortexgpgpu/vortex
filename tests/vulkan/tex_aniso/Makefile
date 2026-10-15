# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Anisotropic filtering for the vortexpipe driver: a footprint 4x longer than
# wide, isotropic against maxAnisotropy 16, on a texture whose answer does not
# depend on where an implementation places its probes.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_aniso

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/tex_aniso.vert \
           $(SRC_DIR)/tex_aniso.frag

include ../common.mk
