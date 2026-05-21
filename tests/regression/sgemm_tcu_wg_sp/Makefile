ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_TCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TCU_ENABLE)

PROJECT := sgemm_tcu_wg_sp

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

CONFIGS := $(if $(findstring -DVX_CFG_NUM_WARPS=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_NUM_WARPS=4)
CONFIGS := $(if $(findstring -DVX_CFG_ISSUE_WIDTH=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_ISSUE_WIDTH=4)
CONFIGS += -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_TCU_SPARSE_ENABLE -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_RS

OPTS := $(OPTS) $(if $(findstring -w,$(OPTS)),,-w 4)

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

KERNEL_LIB := vortex2

include ../common.mk
