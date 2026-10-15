# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Sampler address-mode test for the vortexpipe driver: CLAMP, REPEAT, MIRROR,
# BORDER and MIRROR_CLAMP on the fixed-function texture unit, all against one
# texture so two of them agreeing is a failure.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := texwrap

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/texwrap.vert \
           $(SRC_DIR)/texwrap.frag

include ../common.mk
