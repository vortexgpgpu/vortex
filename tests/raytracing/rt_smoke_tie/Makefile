ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)
# CW-BVH4 scene -> build the RTU as a CW-BVH4 walker.
CONFIGS += -DVX_CFG_RTU_BVH_WIDTH=4

PROJECT := rt_smoke_tie

SRC_DIR := $(VORTEX_HOME)/tests/raytracing/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp
HDRS := $(SRC_DIR)/common.h $(SRC_DIR)/golden.h

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(SRC_DIR)/common.h

OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk
