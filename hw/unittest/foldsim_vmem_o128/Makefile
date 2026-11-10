ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The v8 kernel's reads in flight: 128 on a 1-bit id (64 per id), a 64-line cache.
PROJECT := foldsim_vmem_o128

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(FOLDSIM_SHELL)/rtl

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem

# The controller's parameters, to the RTL and the test alike.
VMEM := AXI_ID_W=1 OUTSTANDING=128 CACHE_ROWS=64

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(FOLDSIM_SHELL)/include
CXXFLAGS += $(foreach p,$(VMEM),-DVM_$(p))

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR)
PARAMS += $(foreach p,$(VMEM),-G$(p))

TOP := foldsim_vmem

include ../common.mk
