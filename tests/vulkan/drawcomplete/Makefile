# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Three draw forms on the vortexpipe device path: primitive restart on a
# triangle list, a pipeline without a fragment shader, and a render pass with
# no attachments.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := drawcomplete

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/drawcomplete.vert \
           $(SRC_DIR)/drawcomplete.frag \
           $(SRC_DIR)/drawcomplete_count.frag

include ../common.mk
