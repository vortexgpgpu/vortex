# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Shared-memory atomic coverage test for the vortexpipe driver.
# atomic_shared.comp contends every integer atomic the translator maps on
# `shared` variables, so each AMO's address lands in the local-memory aperture,
# and the host checks order-independent invariants per workgroup.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run on Vortex via vortexpipe
#
# Not runnable on rtlsim: VX_lmem_switch has no AMO path -- see the suite
# Makefile's EXCLUDE_rtlsim entry.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := atomic_shared

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/atomic_shared.comp

include ../common.mk
