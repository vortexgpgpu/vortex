ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The point of this demo is to show DXA multicast paying off under a
# cluster_dim launch, so DXA has to be enabled.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)

PROJECT := cta_cluster

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

KERNEL_LIB := vortex2

include ../common.mk
