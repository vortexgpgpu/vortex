# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Primitive-order merge test for the vortexpipe driver: sixteen full-screen
# layers blended source-over in one instanced draw, so every pixel merges
# sixteen fragments from different warps and the result depends on their
# order. Runs on RGBA8 (fixed-function output merger) and RGBA16F (software
# merger).
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the case via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := blendorder

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/blendorder.vert \
           $(SRC_DIR)/blendorder.frag

LDLIBS += -lm

include ../common.mk
