# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Oversized compute workgroup with no shared memory -- the shape a driver is
# most tempted to split across smaller device workgroups, and the one where
# doing so silently corrupts gl_GlobalInvocationID.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- dispatch 128-wide workgroups + check the guard region

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := bigwg

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

# Handing this dispatch to llvmpipe is a correct outcome -- the workgroup is
# larger than any Vortex CTA. Strict mode would turn that into a failure, and
# the test is about the answer, not about where it was computed. Set before
# common.mk.
STRICT := 0

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/bigwg.comp

include ../common.mk
