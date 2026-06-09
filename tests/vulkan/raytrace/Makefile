# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Ray-query smoke test for the vortexpipe driver: casts primary rays
# at a one-triangle acceleration structure. The ray tracing runs on
# lavapipe's software path as a correctness oracle.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- trace the rays via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := raytrace

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/raytrace.comp

# rayQueryEXT needs SPIR-V 1.4 -- compile against the Vulkan 1.2 env.
GLSLC := glslc --target-env=vulkan1.2

# This test deliberately runs ray tracing on llvmpipe. The harness default
# (STRICT=1) would refuse the llvmpipe fallback and fail; turn off strict
# mode so the fallback is the EXPECTED path. The device check in common.mk
# still validates that vortexpipe was loaded (the device name carries the
# "vortexpipe (Vortex on …)" prefix because vortexpipe's pipe_screen is in
# front, even when launches fall back).
STRICT := 0

include ../common.mk
