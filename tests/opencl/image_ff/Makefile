ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := image_ff

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

# Route image reads through the fixed-function TEX unit (Tier A). SimX provides
# the SETW/GETW graphics-window ops (that vx_tex4 uses) via the RTU model, so
# enable RTU alongside TEX exactly as tests/graphics/gfx_tex4 does. Without these
# the device reports no TEX and the same test still passes via software sampling.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TEX_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TEX_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE -DVX_CFG_RTU_BVH_WIDTH=0)

SRCS := $(SRC_DIR)/main.cc

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

OPTS ?= -w 16 -h 16

include ../common.mk
