# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# 64-bit atomic load/store tearing test: atomic64_tear.comp mixes
# 64-bit atomic adds with atomic loads and stores on a storage buffer, shared
# memory and a buffer device address; no load may see a torn value.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run on Vortex via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := atomic64_tear

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/atomic64_tear.comp

GLSLC := glslc --target-env=vulkan1.2

STRICT := 1

include ../common.mk
