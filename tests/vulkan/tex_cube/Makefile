# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Cube-map test for the vortexpipe driver: each of the six faces carries a
# distinct colour, sampled by a direction vector pointing at it.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the case via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_cube

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/tex_cube.vert \
           $(SRC_DIR)/cube_faces.frag

include ../common.mk
