ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DEXT_TEX_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DEXT_TEX_ENABLE)

PROJECT := tex_smoke

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

KERNEL_LIB := vortex2

include ../common.mk
