# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# 4x multisample shading rate for the vortexpipe driver: gl_SampleID and
# gl_SamplePosition, minSampleShading = 1.0 evaluating an input at each sample,
# and a centroid input on a partly covered pixel, each read back per sample
# through a sampler2DMS read pass.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := msaa_rate

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/rate.vert \
           $(SRC_DIR)/rate_sid.frag \
           $(SRC_DIR)/rate_minss.frag \
           $(SRC_DIR)/rate_centroid.frag \
           $(SRC_DIR)/rate_read.frag

LDLIBS += -lm

include ../common.mk
