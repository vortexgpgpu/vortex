# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Two textures sampled by one fragment shader, each through its own sampler:
# TEX stage 0 and stage 1 at once.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_stages

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/tex_stages.vert \
           $(SRC_DIR)/tex_stages.frag

include ../common.mk
