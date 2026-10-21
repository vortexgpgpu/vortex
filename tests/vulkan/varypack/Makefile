# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Varyings laid out by location: outputs sharing locations by component next
# to arrays, a struct and a matrix, checked as transform feedback captures them
# and as the fragment stage reads them.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- draw, capture, and check both
#
# STRICT is 1: the draw must run on the device. Set before common.mk.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := varypack

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

STRICT := 1

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/varypack.vert $(SRC_DIR)/varypack.frag

include ../common.mk
