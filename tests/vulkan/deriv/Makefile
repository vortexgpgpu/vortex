# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Screen-space derivatives: dFdx/dFdy must come from the neighbouring lane of
# the pixel quad, on both axes.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- shade, then compare against the derivative the varying
#                      was constructed to have
#
# STRICT stays at the suite default of 1: llvmpipe computes derivatives
# correctly, so a draw that fell back would pass while testing nothing.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk


PROJECT := deriv

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/deriv.vert $(SRC_DIR)/deriv.frag

include ../common.mk
