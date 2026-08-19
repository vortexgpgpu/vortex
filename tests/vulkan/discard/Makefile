# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# discard and demote in the fragment stage: a suppressed fragment must not reach
# the colour attachment or commit a store, while still executing far enough for
# its quad neighbours to shuffle from it.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- draw the three phases and check what each one left behind
#
# STRICT stays at the suite default of 1: llvmpipe discards correctly, so a draw
# that fell back would pass while testing nothing.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk


PROJECT := discard

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/discard.vert \
           $(SRC_DIR)/discard_kill.frag \
           $(SRC_DIR)/discard_demote.frag \
           $(SRC_DIR)/discard_sink.frag \
           $(SRC_DIR)/discard_vswrite.vert \
           $(SRC_DIR)/discard_atomic.frag

include ../common.mk
