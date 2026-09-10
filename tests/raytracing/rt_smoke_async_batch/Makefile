ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)
# Flat triangle-list scene -> build the RTU as a flat walker (true-HW model).
CONFIGS += -DVX_CFG_RTU_BVH_WIDTH=0
# The whole point of the test: four traces issued back to back before any wait.
# Four traces in flight is four slots, and the default pool holds one per core.
CONFIGS += -DVX_CFG_RTU_NUM_SLOTS=4

PROJECT := rt_smoke_async_batch

SRC_DIR := $(VORTEX_HOME)/tests/raytracing/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -n1

KERNEL_LIB := vortex2

include ../common.mk
