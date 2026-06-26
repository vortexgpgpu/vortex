ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

LATENCY ?= 7

PROJECT := fma_unit

RTL_DIR := $(VORTEX_HOME)/hw/rtl
SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -DLATENCY=$(LATENCY)

PARAMS := -GLATENCY=$(LATENCY) -GMAN_BITS=23 -GEXP_BITS=8

CONFIGS += -DSIMULATION

ifdef LEAN
PARAMS   += -GSNORM_ENABLE=0 -GEXCEPT_ENABLE=0
CXXFLAGS += -DLEAN
endif

SRCS += $(SW_COMMON_DIR)/softfloat_ext.cpp
SRCS += $(SW_COMMON_DIR)/rvfloats.cpp
SRCS += $(SRC_DIR)/main.cpp

RTL_PKGS += $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/fpu/VX_fpu_pkg.sv
RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(RTL_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/fpu -I$(SRC_DIR)

# Generated headers (VX_config.vh, VX_types.vh) live in the build tree; add them
# as an include-search path only, not a source-discovery root, so the recursive
# RTL find never descends into hw/syn build artifacts.
VL_FLAGS += -I$(ROOT_DIR)/hw

TOP := VX_fma_unit

include ../common.mk
