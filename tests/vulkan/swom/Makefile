# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# SW output-merger test (P1-7): the multidraw depth-persistence scene built on a
# device WITHOUT the OM unit (NO_OM := 1). vortexpipe compiles the fragment
# shader to run the output-merger in SIMT software (depth test + framebuffer
# write via the LSU) instead of vx_om4, so a correct result proves the on-device
# SW-OM fork — never falling back to the host. Reuses the multidraw sources.
#
#   make run-simx   -- render via vortexpipe with the OM unit absent

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := swom

# Reuse the multidraw scene (two depth-ordered draws); only the device config
# differs (no OM unit).
SRC_DIR := $(VORTEX_HOME)/tests/vulkan/multidraw

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/near.vert $(SRC_DIR)/far.vert $(SRC_DIR)/depth.frag

NO_OM := 1

include ../common.mk
