# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Integer storage-image stores that saturate to each format's range, not
# wrap into their low bits.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- store past each range + check the clamp

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := imgsat

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/imgsat.comp

include ../common.mk
