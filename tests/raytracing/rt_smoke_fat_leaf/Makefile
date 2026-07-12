ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)
# CW-BVH4 walker. The scene root IS a single fat LEAF_TRI packing several
# triangles; the walker must iterate all of them (I7), not just the first.
CONFIGS += -DVX_CFG_RTU_BVH_WIDTH=4

PROJECT := rt_smoke_fat_leaf

SRC_DIR := $(VORTEX_HOME)/tests/raytracing/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk
