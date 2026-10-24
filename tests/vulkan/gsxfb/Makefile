# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# gsxfb -- transform feedback, vertex streams and queries from a geometry
# shader: per-stream capture, overflow judged per stream, run-time counts,
# pause/resume, and the stream and pipeline-statistics queries.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run every case
#
# STRICT stays at 1: llvmpipe captures geometry streams correctly, so a draw
# that fell back would pass without testing the device.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := gsxfb

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/gsx.vert $(SRC_DIR)/gsx.frag $(SRC_DIR)/streams.geom \
           $(SRC_DIR)/strip.geom $(SRC_DIR)/inst.geom $(SRC_DIR)/mixed_strip.geom \
           $(SRC_DIR)/mixed_points.geom

include ../common.mk
