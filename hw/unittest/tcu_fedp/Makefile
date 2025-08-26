ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

NUM_REGS ?= 2
LATENCY  ?= 4

PROJECT := tcu_fedp

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I/$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(ROOT_DIR)/hw/tcu
#CXXFLAGS += -I$(ROOT_DIR)/hw/tcu/drl
#CXXFLAGS += -I$(ROOT_DIR)/hw/tcu/bhf

CXXFLAGS += -DNUM_REGS=$(NUM_REGS) -DLATENCY=$(LATENCY)
PARAMS := -GN=$(NUM_REGS) -GLATENCY=4

VL_FLAGS += -DSV_DPI

SRCS += $(DPI_DIR)/dpi_util.cpp
SRCS += $(DPI_DIR)/dpi_float.cpp
SRCS += $(SW_COMMON_DIR)/softfloat_ext.cpp
SRCS += $(SW_COMMON_DIR)/rvfloats.cpp
SRCS += $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_PKGS :=

RTL_INCLUDE := -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/tcu
RTL_INCLUDE += -I$(RTL_DIR)/tcu/drl
RTL_INCLUDE += -I$(RTL_DIR)/tcu/bhf
RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/hardfloat/source/RISCV
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/fNToRecFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/addRecFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/bsg_hardfloat_pkg.sv
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/HardFloat_primitives.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/HardFloat_rawFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/isSigNaNRecFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/mulRecFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/recFNToFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/recFNToRecFN.v

TOP := VX_tcu_fedp_dpi
ifneq (,$(findstring -DTCU_DRL, $(CONFIGS)))
TOP := VX_tcu_fedp_drl
endif
ifneq (,$(findstring -DTCU_BHF, $(CONFIGS)))
TOP := VX_tcu_fedp_bhf
endif
ifneq (,$(findstring -DTCU_DSP, $(CONFIGS)))
TOP := VX_tcu_fedp_dsp
endif

# test fp16
OPTS ?= --fmt=1 --ext=5 --sig=10 --no-infinities  --no-nans

include ../common.mk
