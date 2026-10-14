# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Early fragment tests on the vortexpipe device path: a shader with side
# effects runs for exactly the fragments the depth/stencil tests let through,
# before it runs when it declares early_fragment_tests and regardless of them
# when it does not.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := earlytests

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/et.vert \
           $(SRC_DIR)/et_late.frag \
           $(SRC_DIR)/et_early.frag \
           $(SRC_DIR)/et_early_discard.frag

include ../common.mk
