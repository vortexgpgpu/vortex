# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Control-flow compute test for the vortexpipe driver (Phase 7, step 2):
# a kernel with a loop + nested branch, exercising the structured
# control flow vp_nir_to_llvm gained in this step.
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
