# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# vkCmdDrawIndirect for the vortexpipe driver: two draws whose parameters come
# from a buffer, each compared against the direct draw of the same triangle.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := drawindirect

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/drawindirect.vert \
           $(SRC_DIR)/drawindirect.frag

include ../common.mk
