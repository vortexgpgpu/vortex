ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := image_ff_linear

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

# Route bilinear image reads through the fixed-function TEX unit (Tier A). As in
# tests/graphics/gfx_tex4, SimX serves the SETW/GETW graphics window via the RTU
# model, so enable RTU alongside TEX. Without these the same test passes via the
# software bilinear sampler.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TEX_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TEX_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE -DVX_CFG_RTU_BVH_WIDTH=0)

SRCS := $(SRC_DIR)/main.cc

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

OPTS ?= -w 16 -h 16

include ../common.mk
