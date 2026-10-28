# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Transfers into and out of partially resident sparse images two or more 64 KiB
# tiles wide: buffer <-> 3D image copies, an image copy between two sparse
# images bound differently, a layer clear and a scaled blit. Each texel must
# land in its own tile, which the test reads through the bound memory.
#
#   make            -- build the host binary
#   make run-simx   -- run every case + check every resident texel

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := sparse_xfer

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS :=

include ../common.mk
