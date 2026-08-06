ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := image_hwsw

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

# Needs the fixed-function TEX unit so the first two images sample in hardware
# while the third overflows to software; RTU provides the SETW/GETW window in
# SimX (as gfx_tex4 does). Without TEX all three sample in software and the test
# trivially passes (hw == sw because both are sw).
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TEX_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TEX_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE -DVX_CFG_RTU_BVH_WIDTH=0)

SRCS := $(SRC_DIR)/main.cc

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

OPTS ?= -w 32 -h 32

include ../common.mk
