# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# 4x multisample with two colour attachments (RGBA8, R32F) and a D32F depth
# attachment for the vortexpipe driver: a sloped triangle crossing a flat quad,
# where which one wins is decided per sample by the depth at that sample.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := msaa_mrt

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/mrt.vert \
           $(SRC_DIR)/mrt.frag \
           $(SRC_DIR)/mrt_quad.vert \
           $(SRC_DIR)/mrt_read.frag

include ../common.mk
