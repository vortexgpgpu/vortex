# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Draws whose primitive restarts cut every run too short for a primitive: the
# device finds that in the restart scan and skips the rest of each draw, so the
# run retires no more instructions than the scans take (iaasm --cut).
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := iaasm

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/iaasm

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/iaasm.vert \
           $(SRC_DIR)/iaasm.frag

OPTS := --cut 64
EXPECT_MAX_INSTRS := 400000

include ../common.mk
