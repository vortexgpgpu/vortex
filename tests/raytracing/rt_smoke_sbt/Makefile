ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)
# Flat triangle-list scene -> build the RTU as a flat walker without TLAS, which
# would expect a TLAS scene (true-HW model; overrides a profile that enables it).
CONFIGS += -DVX_CFG_RTU_BVH_WIDTH=0 -DVX_CFG_RTU_TLAS_DISABLE

PROJECT := rt_smoke_sbt

SRC_DIR := $(VORTEX_HOME)/tests/raytracing/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk
