# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# geometry -- geometry shaders on the device: pass-through, strips, expansion,
# point size and polygon mode, each compared pixel for pixel with the image the
# specification gives.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run every case
#
# STRICT stays at the suite default of 1: llvmpipe runs geometry shaders
# correctly, so a draw that fell back would pass without testing the device.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := geometry

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/gs.vert $(SRC_DIR)/flat.frag $(SRC_DIR)/pass_tri.geom \
           $(SRC_DIR)/pass_line.geom $(SRC_DIR)/pass_point.geom $(SRC_DIR)/strip.geom \
           $(SRC_DIR)/quad.geom $(SRC_DIR)/dyn.geom $(SRC_DIR)/dynstrip.geom \
           $(SRC_DIR)/dynpoint.geom $(SRC_DIR)/inst1.geom $(SRC_DIR)/inst2.geom \
           $(SRC_DIR)/inst8.geom $(SRC_DIR)/inst32.geom $(SRC_DIR)/adj_lines.geom \
           $(SRC_DIR)/adj_tris.geom $(SRC_DIR)/primid.vert $(SRC_DIR)/primid.frag \
           $(SRC_DIR)/primid.geom

include ../common.mk
