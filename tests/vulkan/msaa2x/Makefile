# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# The multisample counts the driver offers must be the ones its rasterizer can
# place. 2x is not advertised -- merging a 4-sample coverage mask at 2x would
# put the coverage in the wrong places -- so an application cannot ask for it,
# and this pins the advertised set so widening it stays a deliberate act.
#
# No shaders: the contract is the reported limit, and rendering nothing is what
# an application that honours the limit does with 2x.
#
#   make            -- build the host binary
#   make run-simx   -- run the check via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := msaa2x

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS :=

include ../common.mk
