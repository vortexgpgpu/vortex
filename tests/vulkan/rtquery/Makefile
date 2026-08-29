# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# VK_KHR_ray_query test that actually runs on the PRISM RTU. Same scene as
# tests/vulkan/raytrace, but a 4x4=16-thread workgroup (fits one Vortex CTA
# at the default 4 threads x 4 warps) so the ray-query compute shader is
# dispatched on the Vortex cores instead of falling back to llvmpipe. The
# rq_* intrinsics are lowered to RTU vx_rt_* ops (vp_nir_lower_ray_tracing_
# to_rtu), and the host AS is transcoded to a CW-BVH4 scene (vp_launch.c);
# the RTU walker resolves the hits.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- trace the rays on the RTU via vortexpipe
#
# NOTE: requires an RTU-enabled mesa-vortex install (pass MESA_PATH=<rtu
# prefix>); the RTU simx is selected by the EXT_RTU_ENABLE config below.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Build the simx runtime with the RTU extension so the device advertises
# VX_ISA_EXT_RTU; without it vortexpipe sees has_rtu=0 and lavapipe expands
# the ray query to a software BVH walk that falls back to llvmpipe.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)

PROJECT := rtquery

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/rtquery.comp

# rayQueryEXT needs SPIR-V 1.4 -- compile against the Vulkan 1.2 env.
GLSLC := glslc --target-env=vulkan1.2

# The ray-query compute shader runs on the RTU; lavapipe's internal
# acceleration-structure build shaders (subgroup/atomic ops) still run on
# llvmpipe, so STRICT must stay 0 (a global refuse-fallback would trip on
# those builders). The device check in common.mk validates vortexpipe is
# the active driver.
STRICT := 0

include ../common.mk
