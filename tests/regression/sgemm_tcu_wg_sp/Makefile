ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DEXT_TCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DEXT_TCU_ENABLE)

PROJECT := sgemm_tcu_wg_sp

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

CONFIGS := $(if $(findstring -DNUM_WARPS=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DNUM_WARPS=8)
CONFIGS := $(if $(findstring -DISSUE_WIDTH=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DISSUE_WIDTH=4)
CONFIGS += -DTCU_WGMMA_ENABLE -DTCU_SPARSE_ENABLE -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_RS

OPTS := $(OPTS) $(if $(findstring -w,$(OPTS)),,-w 4)

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

KERNEL_LIB := vortex2

include ../common.mk
