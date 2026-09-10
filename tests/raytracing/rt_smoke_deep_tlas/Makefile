ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)
# CW-BVH4 TLAS with a DELIBERATELY shallow short stack. Every instance's BLAS is
# deep enough to overflow, so each descent must restart. The per-BLAS
# restart budget must be reset on every instance entry, otherwise a deep
# FIRST instance exhausts one global budget and the (closest-hit) LAST instance
# drops subtrees and returns a farther hit.
CONFIGS += -DVX_CFG_RTU_BVH_WIDTH=4 -DVX_CFG_RTU_STACK_DEPTH=3

PROJECT := rt_smoke_deep_tlas

SRC_DIR := $(VORTEX_HOME)/tests/raytracing/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk
