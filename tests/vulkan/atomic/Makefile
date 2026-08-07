# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Atomic coverage test for the vortexpipe driver. atomic.comp contends every
# integer atomic the translator maps, on both the SSBO and the shared address
# spaces, and the host checks order-independent invariants over the results.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run on Vortex via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := atomic

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/atomic.comp

include ../common.mk
