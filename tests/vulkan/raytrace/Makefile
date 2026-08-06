# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Shaded VK_KHR_ray_query render that runs 100% on the PRISM RTU. Three
# overlapping opaque triangles at increasing depth are cast at by one
# orthographic ray per pixel; a 4x4=16-thread workgroup (one Vortex CTA
# at the default 4 threads x 4 warps) keeps the dispatch on the Vortex
# cores. The rq_* intrinsics lower to RTU vx_rt_* ops (vp_nir_lower_ray_
# tracing_to_rtu) and the host AS is transcoded to a CW-BVH6 scene
# (vp_launch.c); the RTU walker resolves the front-most hit, which the
# shader colours by gl_PrimitiveID.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- trace + shade the rays on the RTU via vortexpipe
#
# NOTE: requires an RTU-enabled mesa-vortex install (pass MESA_PATH=<rtu
# prefix>); the RTU simx is selected by the EXT_RTU_ENABLE config below.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Build the simx runtime with the RTU extension so the device advertises
# VX_ISA_EXT_RTU; without it vortexpipe sees has_rtu=0 and lavapipe expands
# the ray query to a software BVH walk that falls back to llvmpipe.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)

# vortexpipe transcodes the acceleration structure to a CW-BVH4 scene
# (scene_kind=2), so the RTU must be built width-4 to decode it.
CONFIGS := $(if $(findstring VX_CFG_RTU_BVH_WIDTH,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_RTU_BVH_WIDTH=4)

PROJECT := raytrace

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/raytrace.comp

# rayQueryEXT needs SPIR-V 1.4 -- compile against the Vulkan 1.2 env.
GLSLC := glslc --target-env=vulkan1.2

# The ray-query compute shader runs on the RTU; lavapipe's internal
# acceleration-structure build shaders (subgroup/atomic ops) still run on
# llvmpipe, so STRICT must stay 0 (a global refuse-fallback would trip on
# those builders). The device check in common.mk validates vortexpipe is
# the active driver.
STRICT := 0

include ../common.mk
