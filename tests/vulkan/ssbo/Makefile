# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# SSBO write-readback + SSBO atomic test (W7) for the vortexpipe driver:
# a compute shader writes computed values into one storage buffer and
# atomically accumulates into another; the host reads both back and
# checks the array AND the atomic counter.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run on Vortex via vortexpipe

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := ssbo

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/ssbo.comp

# SSBO atomics lower onto the RISC-V A-extension (amoadd.w / LR.W-SC.W), so the
# device build must enable it — same test-level CONFIGS knob common.mk uses for
# the RASTER/TEX/OM fixed-function units below. Without it the LSU aborts on the
# first AMO ("unsupported in this build").
CONFIGS += -DVX_CFG_EXT_A_ENABLE

include ../common.mk
