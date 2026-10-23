# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# 4x multisample coverage controls for the vortexpipe driver: alpha-to-coverage,
# the pipeline sample mask, a shader-written gl_SampleMask and alpha-to-one,
# each observed per sample through a sampler2DMS read pass.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := msaa_cov

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/cov.vert \
           $(SRC_DIR)/cov_a2c.frag \
           $(SRC_DIR)/cov_solid.frag \
           $(SRC_DIR)/cov_smask.frag \
           $(SRC_DIR)/cov_a2one.frag \
           $(SRC_DIR)/cov_read.frag

LDLIBS += -lm

FF_RASTER := 1
FF_OM := 1

include ../common.mk
