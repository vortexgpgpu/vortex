# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Multiple-render-target (MRT) Vulkan test for the vortexpipe driver
# (issue I4): the whole VS->raster->OM draw runs on the Vortex device
# and the fragment shader writes EIGHT colour attachments, each a
# distinct computed value. The host reads back all eight and checks
# each one (a narrower wiring would leave the rest at clear).
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render + verify via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := mrt

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/mrt.vert $(SRC_DIR)/mrt.frag

include ../common.mk
