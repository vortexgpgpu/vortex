ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_TFU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TFU_ENABLE)

PROJECT := tfu

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/tfu.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_HDRS := $(VORTEX_HOME)/sw/kernel/include/vx_tfu.h $(SRC_DIR)/common.h

OPTS ?= -n4096

KERNEL_LIB := vortex2

include ../common.mk
