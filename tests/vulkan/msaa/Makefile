# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# 4x multisample + resolve test for the vortexpipe driver: the merger keeps
# one colour per sample and the pass-end resolve averages them.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := msaa

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/msaa.vert \
           $(SRC_DIR)/msaa_tri.vert \
           $(SRC_DIR)/msaa.frag

include ../common.mk
