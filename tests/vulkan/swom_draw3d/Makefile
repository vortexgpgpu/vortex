# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# SW output-merger bit-exact test (P1-7): the draw3d scene built on a device
# WITHOUT the OM unit (NO_OM := 1). vortexpipe runs the output-merger in SIMT
# software; the rendered PNG must be BINARY-EQUAL to the same golden reference
# the HW-OM path matches — proving the SW OM is colour-correct, not just
# depth-ordering correct. Reuses the draw3d sources + assets.
#
#   make run-simx   -- replay triangle.cgltrace via vortexpipe, OM unit absent

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := swom_draw3d

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/draw3d

SRCS := $(SRC_DIR)/main.cpp

SHADERS := $(SRC_DIR)/draw3d.vert $(SRC_DIR)/draw3d.frag

CC          := g++
THIRD_PARTY := $(VORTEX_HOME)/third_party
CFLAGS  += -std=c++17 -I$(THIRD_PARTY) -I$(THIRD_PARTY)/cocogfx/include
LDLIBS  += $(THIRD_PARTY)/cocogfx/libcocogfx.a -lpng -lz -lboost_serialization

DRAW3D_ASSETS := $(VORTEX_HOME)/tests/graphics/gfx_draw3d
OPTS ?= -t $(DRAW3D_ASSETS)/triangle.cgltrace \
        -r $(DRAW3D_ASSETS)/triangle_ref_128.png -w 128 -h 128

NO_OM := 1

include ../common.mk
