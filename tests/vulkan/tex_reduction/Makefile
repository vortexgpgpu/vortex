# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Sampler min/max reduction for the vortexpipe driver: three LINEAR probes of a
# 4x4 texture under MIN and MAX, each over exactly its nonzero-weight taps.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_reduction

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/tex_reduction.vert \
           $(SRC_DIR)/tex_reduction.frag

include ../common.mk
