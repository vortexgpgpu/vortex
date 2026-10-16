# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Precise occlusion queries around draws the device runs, against a depth
# buffer, on the fixed-function merger (RGBA8) and on the software merger (4x
# MSAA, RGBA32F). The device counts the samples that pass the depth test, so a
# draw whose count never reaches the query leaves it short.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render each configuration + check the query results
#
# STRICT is 1: the draw must stay on the device while the query is open. Set
# before common.mk.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := occlusion

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

STRICT := 1

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/occlusion.vert $(SRC_DIR)/occlusion.frag

include ../common.mk
