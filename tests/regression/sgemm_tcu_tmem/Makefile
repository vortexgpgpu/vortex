ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_TCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TCU_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)

PROJECT := sgemm_tcu_tmem

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

CONFIGS := $(if $(findstring -DVX_CFG_NUM_WARPS=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_NUM_WARPS=4)
CONFIGS := $(if $(findstring -DVX_CFG_ISSUE_WIDTH=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_ISSUE_WIDTH=4)
CONFIGS := $(if $(findstring -DVX_CFG_TCU_WGMMA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_TCU_WGMMA_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_TCU_FEDP2K,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_TCU_FEDP2K)
CONFIGS := $(if $(findstring -DVX_CFG_TCU_TMEM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_TCU_TMEM_ENABLE)
CONFIGS := $(if $(findstring -DDEEP_BUF,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DDEEP_BUF)
CONFIGS += $(if $(PIPE_N),-DPIPE_N=$(PIPE_N))
CONFIGS += $(if $(TMEM_CLOSED),-DTMEM_CLOSED)
CONFIGS += $(if $(TMEM_PERSTAGE),-DTMEM_PERSTAGE)
CONFIGS += $(if $(TMEM_LANE_PER_THREAD),-DTMEM_LANE_PER_THREAD)


CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

KERNEL_LIB := vortex2

include ../common.mk
