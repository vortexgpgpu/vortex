# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Non-power-of-two texture addressing for the vortexpipe driver: a 5x3 texture
# probed on both axes and through REPEAT, which the fixed-function unit cannot
# address by shifting and so has to reach the software sampler.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := texnpot

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/texnpot.vert \
           $(SRC_DIR)/texnpot.frag

include ../common.mk
