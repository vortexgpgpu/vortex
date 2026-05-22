# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Shared-memory compute test for the vortexpipe driver (Phase 7, step 3):
# a workgroup tree reduction in shared memory, exercising the shared
# memory + barrier support vp_nir_to_llvm gained in this step.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the kernel via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := shmem

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/shmem.comp

include ../common.mk
