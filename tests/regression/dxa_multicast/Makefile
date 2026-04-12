ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DEXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DEXT_DXA_ENABLE)

PROJECT := dxa_multicast

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -r16 -c16 -R16 -C4

KERNEL_LIB := vortex2

include ../common.mk
