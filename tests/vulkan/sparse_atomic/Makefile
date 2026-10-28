# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Contended float adds and exchanges on 2D and 3D sparse R32_SFLOAT images
# several 64 KiB tiles wide, each accumulator in a tile of its own.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run + check every accumulator and exchange

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := sparse_atomic

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/sa.comp

GLSLC := glslc --target-env=vulkan1.2

include ../common.mk
