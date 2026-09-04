ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dxa_s2g_bulk_groups_ws
SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

override CONFIGS += -DVX_CFG_EXT_DXA_ENABLE \
                    -DVX_CFG_EXT_DXA_GROUP_ENABLE \
                    -DVX_CFG_EXT_DXA_S2G_ENABLE \
                    -DVX_CFG_NUM_WARPS=8 \
                    -DVX_CFG_NUM_DXA_CORES=2 \
                    -DVX_CFG_DXA_GROUP_CONTEXTS=64

SRCS := $(SRC_DIR)/main.cpp
VX_SRCS := $(SRC_DIR)/kernel.cpp
KERNEL_LIB := vortex2

include ../common.mk
