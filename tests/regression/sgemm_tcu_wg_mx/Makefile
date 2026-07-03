ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_TCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TCU_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_TCU_MX_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_TCU_MX_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_NUM_WARPS=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_NUM_WARPS=4)
CONFIGS := $(if $(findstring -DVX_CFG_ISSUE_WIDTH=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_ISSUE_WIDTH=4)
CONFIGS += -DVX_CFG_TCU_WGMMA_ENABLE
CONFIGS := $(if $(findstring -DWGMMA_SS,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DWGMMA_RS)

PROJECT := sgemm_tcu_wg_mx
SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)
SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp
VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS := $(OPTS) $(if $(findstring -m,$(OPTS)),,-m 64)
OPTS := $(OPTS) $(if $(findstring -n,$(OPTS)),,-n 64)
OPTS := $(OPTS) $(if $(findstring -k,$(OPTS)),,-k 64)
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a
KERNEL_LIB := vortex2

include ../common.mk
