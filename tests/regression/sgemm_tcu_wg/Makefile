ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DEXT_TCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DEXT_TCU_ENABLE)

PROJECT := sgemm_tcu_wg

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

CONFIGS := $(if $(findstring -DNUM_WARPS=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DNUM_WARPS=4)
CONFIGS := $(if $(findstring -DISSUE_WIDTH=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DISSUE_WIDTH=4)
CONFIGS += -DTCU_WGMMA_ENABLE -DWGMMA_RS

# RTLsim is roughly two orders of magnitude slower than SimX; keep the
# default matrix size small enough for a 60s sweep budget. Override
# -m/-n/-k at the command line for full-size correctness sweeps.
OPTS := $(OPTS) $(if $(findstring -w,$(OPTS)),,-w 4)
OPTS := $(OPTS) $(if $(findstring -m,$(OPTS)),,-m 16)
OPTS := $(OPTS) $(if $(findstring -n,$(OPTS)),,-n 16)
OPTS := $(OPTS) $(if $(findstring -k,$(OPTS)),,-k 16)

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

KERNEL_LIB := vortex2

include ../common.mk
