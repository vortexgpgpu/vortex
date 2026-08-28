# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# A multiview render pass replays its one draw once per view, each into its own
# attachment layer. A driver that renders it once leaves every view but the
# first at the clear colour, which this checks layer by layer.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render the pass + check every view
#
# STRICT is 0 because refusing the pass and rendering it on llvmpipe is a
# correct outcome; the assertion is the pixels, not where they were produced.
# Set before common.mk.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := multiview

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

STRICT := 0

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/multiview.vert $(SRC_DIR)/multiview.frag

include ../common.mk
