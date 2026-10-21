# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Viewport arrays, per-viewport scissors, gl_ViewportIndex and gl_Layer for the
# vortexpipe driver, single-sample (fixed-function rasterizer), 4x multisample
# (software rasterizer) and layered; per-viewport depth clipping and clamping,
# out-of-range indices, lines and points, and layered depth with blending.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases via vortexpipe on SimX
#
# STRICT stays at the suite default of 1: llvmpipe renders all of this
# correctly, so only a refused fallback shows the device did the work.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := multiviewport

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/mvp_viewport.vert \
           $(SRC_DIR)/mvp_layer.vert \
           $(SRC_DIR)/mvp.frag \
           $(SRC_DIR)/mvp_attr.vert

include ../common.mk

# Vulkan 1.2 SPIR-V: the vertex shaders write gl_ViewportIndex and gl_Layer
# under the ShaderViewportIndex and ShaderLayer capabilities.
mvp_viewport.vert.spv mvp_layer.vert.spv mvp_attr.vert.spv: %.spv: $(SRC_DIR)/%
	$(GLSLC) --target-env=vulkan1.2 $< -o $@

# glslang declares the Geometry capability for gl_Layer in a fragment shader,
# which this device does not offer; ShaderLayer (shaderOutputLayer) also allows
# it, so the module is reassembled with that capability instead.
mvp.frag.spv: $(SRC_DIR)/mvp.frag
	$(GLSLC) --target-env=vulkan1.2 $< -o - | spirv-dis - | \
	   sed 's/OpCapability Geometry/OpCapability ShaderLayer/' | \
	   spirv-as --target-env vulkan1.2 - -o $@
