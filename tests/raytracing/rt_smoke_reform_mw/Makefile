ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)
# Flat triangle-list scene -> build the RTU as a flat walker (true-HW model).
CONFIGS += -DVX_CFG_RTU_BVH_WIDTH=0
# Reformation batches candidates from several warps into one callback dispatch,
# so several warps have to be tracing AT THE SAME TIME. One slot per core would
# serialise them and the test would pass without ever exercising what it is for.
CONFIGS += -DVX_CFG_RTU_NUM_SLOTS=4

PROJECT := rt_smoke_reform_mw

SRC_DIR := $(VORTEX_HOME)/tests/raytracing/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -w2 -n4

KERNEL_LIB := vortex2

include ../common.mk
