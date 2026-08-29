# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Control-flow compute test for the vortexpipe driver:
# kernel with a loop + nested branch, exercising structured control flow.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the kernel via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := cflow

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/cflow.comp

include ../common.mk
