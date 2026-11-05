ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# 8-byte rows with byte enables, eight to a 64-byte line (FireSim's DRAM words); one AXI id bit, 32 reads in flight.
PROJECT := foldsim_vmem_rb8

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem

# The controller's parameters, to the RTL and the test alike.
VMEM := DW=64 MW=8 IW=4 AW=8 ROW_BYTES=8 AXI_ID_W=1 OUTSTANDING=32

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common
CXXFLAGS += $(foreach p,$(VMEM),-DVM_$(p))

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR)
PARAMS += $(foreach p,$(VMEM),-G$(p))

TOP := foldsim_vmem

include ../common.mk
