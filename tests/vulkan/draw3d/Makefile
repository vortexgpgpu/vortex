# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Vulkan port of the draw3d regression test: replays a CGLTrace scene
# through the vortexpipe driver. The whole graphics pipeline -- vertex
# stage (with vertex-buffer input), RASTER, fragment stage, OM and TEX
# -- runs on the Vortex device.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- replay triangle.cgltrace via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := draw3d

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

SHADERS := $(SRC_DIR)/draw3d.vert $(SRC_DIR)/draw3d.frag

# draw3d replays a CGLTrace, so the host program is C++ and links
# cocogfx (trace load + PNG save / reference compare).
CC          := g++
THIRD_PARTY := $(VORTEX_HOME)/third_party
CFLAGS  += -std=c++17 -I$(THIRD_PARTY) -I$(THIRD_PARTY)/cocogfx/include
LDLIBS  += $(THIRD_PARTY)/cocogfx/libcocogfx.a -lpng -lz -lboost_serialization

# The scene + reference images are shared with the native draw3d test;
# the rendered PNG must be binary-equal to the draw3d reference.
DRAW3D_ASSETS := $(VORTEX_HOME)/tests/regression/gfx_draw3d
OPTS ?= -t $(DRAW3D_ASSETS)/triangle.cgltrace \
        -r $(DRAW3D_ASSETS)/triangle_ref_128.png -w 128 -h 128

include ../common.mk
