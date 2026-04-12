ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DNUM_CORES,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DNUM_CORES=2)

PROJECT := async_gbarrier

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

KERNEL_LIB := vortex2

include ../common.mk
