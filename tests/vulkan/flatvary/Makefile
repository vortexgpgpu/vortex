# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# flat varyings -- integer and float -- carried from the provoking vertex to the
# fragment shader without passing through the interpolator.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- draw one triangle, sample near each of its three vertices
#
# STRICT stays at the suite default of 1 on purpose, and here it is load-bearing
# twice over: llvmpipe carries flat varyings correctly, so a fallback would pass
# without testing anything, and the driver used to REFUSE a flat input outright.
# A run that falls back is the old behaviour, not a pass.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := flatvary

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/flatvary.vert $(SRC_DIR)/flatvary.frag

include ../common.mk
