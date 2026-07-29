ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := smem_amo_mlp
SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)

SRCS := $(SRC_DIR)/main.cpp
VX_SRCS := $(SRC_DIR)/kernel.cpp
OPTS ?= -m0 -n16
KERNEL_LIB := vortex2

include ../common.mk
