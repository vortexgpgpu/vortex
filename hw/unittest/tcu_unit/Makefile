ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tcu_unit

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(ROOT_DIR)/hw

CONFIGS += -DSIMULATION -DSV_DPI -DEXT_TCU_ENABLE

XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py --config $(VORTEX_HOME)/hw/VX_config.toml --cflags '$(CONFIGS)')

SRCS += $(DPI_DIR)/dpi_util.cpp
SRCS += $(DPI_DIR)/dpi_float.cpp
SRCS += $(SW_COMMON_DIR)/softfloat_ext.cpp
SRCS += $(SW_COMMON_DIR)/rvfloats.cpp
SRCS += $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS := -DDBG_TRACE_TCU

RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/tcu/VX_tcu_pkg.sv

RTL_INCLUDE := -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/tcu -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(SRC_DIR)

ifneq (,$(filter -DTCU_TYPE_DPI, $(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/dpi
endif

# default test: fp16
OPTS ?= --fmt=1

TOP := VX_tcu_unit_top

include ../common.mk
