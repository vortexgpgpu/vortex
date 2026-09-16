ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_TCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TCU_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_NUM_THREADS,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_NUM_THREADS=8)

PROJECT := flashattention_tcu

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

KERNEL_LIB := vortex2

include ../common.mk
