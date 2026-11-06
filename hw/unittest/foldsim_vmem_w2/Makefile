ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Two writes on AXI at once, a two-entry hazard table, 7-bit write numbers (wrapping): no cache, so every read of a row written lately meets the table.
PROJECT := foldsim_vmem_w2

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem

# The controller's parameters, to the RTL and the test alike.
VMEM := WRITES_OUT=2 HAZARD_BITS=1 SEQ_W=7

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common
CXXFLAGS += $(foreach p,$(VMEM),-DVM_$(p))

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR)
PARAMS += $(foreach p,$(VMEM),-G$(p))

TOP := foldsim_vmem

include ../common.mk
