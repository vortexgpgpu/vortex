# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# A multisample attachment read directly rather than resolved, through a
# sampler2DMS -- the only way an application can observe it, since a
# multisample image cannot be the source of a copy.
#
# STRICT := 0 because a refusal that hands the pass to llvmpipe is one of the
# two correct outcomes here; refusing the fallback would fail the test for being
# right. What is asserted is that the target comes back one colour and a legal
# one, which is what separates "cleanly refused" from "silently wrong".
STRICT := 0

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := msaa_read

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/msaa_read.vert \
           $(SRC_DIR)/msaa_fill.frag \
           $(SRC_DIR)/msaa_fetch.frag

include ../common.mk
