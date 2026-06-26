ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

ifeq ($(DXA_COPY_FORCE_LSU),1)
CONFIGS := $(filter-out -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS))
else
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)
endif

PROJECT := dxa_copy

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -d2

KERNEL_LIB := vortex2

include ../common.mk
