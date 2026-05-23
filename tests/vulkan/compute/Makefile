# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Vulkan compute smoke test for the vortexpipe driver: add1.comp
# (data[i] += 1) runs through lavapipe + vortexpipe on the Vortex
# SimX device, and the result is checked on the host.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run on Vortex via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := compute

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/add1.comp

include ../common.mk
