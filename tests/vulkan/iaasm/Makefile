# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Input assembly on the device: indexed draws of every index width, strips,
# fans and lists cut by primitive restart, instancing, both provoking-vertex
# modes and back-face culling, each checked against the same primitives drawn
# as a plain list; and the statistics a restart draw counts.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := iaasm

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/iaasm.vert \
           $(SRC_DIR)/iaasm.frag

include ../common.mk
