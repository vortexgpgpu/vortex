# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# gl_ViewIndex for the vortexpipe driver: three views of one multiview pass,
# told apart only by the index the fragment shader reads.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := viewindex

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/viewindex.vert \
           $(SRC_DIR)/viewindex.frag

include ../common.mk
