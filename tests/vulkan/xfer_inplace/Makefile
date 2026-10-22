# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# A chain of transfers through images with padded rows -- buffer to image, image
# to image, a scaled blit, a sub-box clear -- then the result sampled by a
# shader and copied back out, with a different pattern in every case.
#
#   make            -- build the host binary + SPIR-V
#   make run-simx   -- run the cases on SimX

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := xfer_inplace

SRC_DIR := $(VORTEX_HOME)/tests/vulkan/$(PROJECT)

SRCS := $(SRC_DIR)/main.c

SHADERS := $(SRC_DIR)/sample.comp

include ../common.mk
