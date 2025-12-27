ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

NUM_REGS ?= 2
LATENCY  ?= 4

PROJECT := tcu_fedp

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py --config $(VORTEX_HOME)/hw/VX_config.toml --cflags '$(CONFIGS)')

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I/$(THIRD_PARTY_DIR)/softfloat/source/include

CXXFLAGS += -DNUM_REGS=$(NUM_REGS) -DLATENCY=$(LATENCY)
PARAMS := -GN=$(NUM_REGS) -GLATENCY=$(LATENCY)

VL_FLAGS += -DSV_DPI

SRCS += $(DPI_DIR)/dpi_util.cpp
SRCS += $(DPI_DIR)/dpi_float.cpp
SRCS += $(SW_COMMON_DIR)/softfloat_ext.cpp
SRCS += $(SW_COMMON_DIR)/rvfloats.cpp
SRCS += $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/tcu

ifneq (,$(filter -DTCU_TYPE_DPI, $(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/dpi
TOP := VX_tcu_fedp_dpi
endif

ifneq (,$(filter -DTCU_TYPE_DSP, $(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/dsp
TOP := VX_tcu_fedp_dsp
endif

ifneq (,$(filter -DTCU_TYPE_BHF, $(XCONFIGS)))
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/fNToRecFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/addRecFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/bsg_hardfloat_pkg.sv
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/HardFloat_primitives.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/HardFloat_rawFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/isSigNaNRecFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/mulRecFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/recFNToFN.v
RTL_PKGS += $(THIRD_PARTY_DIR)/hardfloat/source/recFNToRecFN.v
RTL_INCLUDE += -I$(RTL_DIR)/tcu/bhf
RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/hardfloat/source/RISCV
TOP := VX_tcu_fedp_bhf
endif

ifneq (,$(filter -DTCU_TYPE_DRL, $(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/drl
TOP := VX_tcu_fedp_drl
endif

# test fp16
OPTS ?= --fmt=1 --ext=5 --sig=10 --no-infinities  --no-nans

include ../common.mk
