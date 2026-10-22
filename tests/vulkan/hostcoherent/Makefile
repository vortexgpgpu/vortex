# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Host-coherent memory across submissions: the host rewrites a persistently
# mapped buffer between submissions with no flush, the device reads it, writes
# it, and the host reads the result after the fence -- with an llvmpipe
# dispatch between two device ones in every other submission.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the six submissions and check every element

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := hostcoherent

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

# The middle dispatch of the odd submissions must really run on llvmpipe, which
# writes the buffer behind the driver's back; strict mode would turn it into a
# no-op. Set before common.mk. The other nine dispatches stay on the device.
STRICT := 0
EXPECT_ATTRIB := launches=9/12_draws=0/0
EXPECT_CPU := workgroup-size=3

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/mad.comp $(SRC_DIR)/inc_shared.comp

include ../common.mk
