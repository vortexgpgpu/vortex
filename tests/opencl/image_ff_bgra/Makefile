ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := image_ff_bgra

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

# Route image reads through the fixed-function TEX unit. TEX takes u/v/lod in
# registers, so no other extension is required.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TEX_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TEX_ENABLE)

SRCS := $(SRC_DIR)/main.cc

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

OPTS ?= -w 16 -h 16

include ../common.mk
