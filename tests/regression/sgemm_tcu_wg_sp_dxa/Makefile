ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DEXT_TCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DEXT_TCU_ENABLE)
CONFIGS := $(if $(findstring -DEXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DEXT_DXA_ENABLE)

PROJECT := sgemm_tcu_wg_sp_dxa

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

CONFIGS := $(if $(findstring -DTCU_WGMMA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DTCU_WGMMA_ENABLE)
CONFIGS := $(if $(findstring -DTCU_SPARSE_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DTCU_SPARSE_ENABLE)
CONFIGS := $(if $(findstring -DITYPE=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DITYPE=fp16)
CONFIGS := $(if $(findstring -DOTYPE=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DOTYPE=fp32)
CONFIGS := $(if $(findstring -DWGMMA_NRC=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DWGMMA_NRC=8)

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

KERNEL_LIB := vortex2

include ../common.mk
