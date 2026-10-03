ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The scene format the RTU walks is a build-time choice (VX_CFG_RTU_BVH_WIDTH):
# replay against the configuration that recorded the log, so CONFIGS is used
# as given, plus the RTU itself.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)

PROJECT := rt_replay

SRC_DIR := $(VORTEX_HOME)/tests/raytracing/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp
HDRS := $(SRC_DIR)/common.h

VX_SRCS := $(SRC_DIR)/kernel.cpp
VX_HDRS := $(SRC_DIR)/common.h

OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk
