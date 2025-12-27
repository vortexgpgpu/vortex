ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := core_top

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I/$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(ROOT_DIR)/hw

VL_FLAGS += -DSV_DPI

SRCS := $(DPI_DIR)/dpi_util.cpp
SRCS += $(DPI_DIR)/dpi_float.cpp
SRCS += $(SW_COMMON_DIR)/softfloat_ext.cpp
SRCS += $(SW_COMMON_DIR)/rvfloats.cpp
SRCS += $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS := -DDBG_TRACE_CACHE

RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/fpu/VX_fpu_pkg.sv

RTL_INCLUDE := -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/mem -I$(RTL_DIR)/fpu -I$(RTL_DIR)/core

TOP := VX_core_top

include ../common.mk