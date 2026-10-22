# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Render into one level and layer of an array image whose rows carry padding,
# with each depth/stencil format the output merger stores, and check every
# texel of the target and of its neighbouring subresources.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run every format on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := rtinplace

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/rtinplace.vert $(SRC_DIR)/rtinplace.frag

# Every format, D32F and D32F_S8 and S8 included, merges on the fixed-function
# output merger, which stores them in place.
FF_OM := 1

include ../common.mk
