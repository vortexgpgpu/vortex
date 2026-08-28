# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# ALU coverage test for the vortexpipe driver: alu.comp runs the
# bitfield / carry-borrow / pack-unpack ops on the Vortex device,
# and every result is reproduced on the host.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run on Vortex via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := alu

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/alu.comp

include ../common.mk
