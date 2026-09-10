# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Destroy-and-replace test for the vortexpipe driver: render, destroy the
# vertex buffer, allocate another of the same size at what is very likely the
# same host address, and render again from different data. Catches a driver
# that caches a device buffer by host pointer without evicting it when the
# resource dies.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render both passes + check the second reflects its own data

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := realloc

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/realloc.vert $(SRC_DIR)/realloc.frag

include ../common.mk
