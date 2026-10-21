# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# VK_EXT_transform_feedback from the vertex stage: capture into two buffers
# with holes over every topology and provoking-vertex mode, primitive restart
# and instancing; overflow and mixed primitive sizes; pause and resume through
# counter buffers across command buffers; capture under rasterizer discard;
# and the stream query beside a primitives-generated query.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- capture + check every byte of the buffers
#
# STRICT is 1: every capturing draw must run on the device. Set before
# common.mk.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := xfb

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

STRICT := 1

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/xfb.vert $(SRC_DIR)/xfb.frag

include ../common.mk
