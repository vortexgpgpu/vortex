ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)
# CW-BVH4 walker with a deliberately shallow short stack so a deep tree
# overflows and exercises the §8.5.1 re-descend-from-root restart.
CONFIGS += -DVX_CFG_RTU_BVH_WIDTH=4 -DVX_CFG_RTU_STACK_DEPTH=4

PROJECT := rtu_smoke_deep_stack

SRC_DIR := $(VORTEX_HOME)/tests/raytracing/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk
