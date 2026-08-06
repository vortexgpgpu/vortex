# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Face-cull Vulkan test for the vortexpipe driver: renders one
# front-facing triangle twice, once with cullMode=BACK (must draw) and
# once with cullMode=FRONT (must be culled), proving vortexpipe wires
# the rasterizer cull state through to the device front end.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- render + check both cull passes via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := cull

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/cull.vert $(SRC_DIR)/cull.frag

include ../common.mk
