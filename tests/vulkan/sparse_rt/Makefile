# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Rendering into sparse colour and depth attachments one 64 KiB tile wide,
# which the output merger addresses in place, and three tiles wide, which only
# the software merger addresses through their tiles. Every pixel, its depth and
# its place in its tile are checked.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run both passes + check every pixel

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := sparse_rt

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/sr.vert $(SRC_DIR)/sr.frag

include ../common.mk
