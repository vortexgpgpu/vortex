ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Four writes on AXI at once, a four-entry hazard table, 7-bit write numbers, behind a 64-line cache (evicted lines on AXI), write first.
PROJECT := foldsim_vmem_w4c64

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem

# The controller's parameters, to the RTL and the test alike.
VMEM := WRITES_OUT=4 HAZARD_BITS=2 SEQ_W=7 CACHE_ROWS=64 RDW_READ_FIRST=0

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common
CXXFLAGS += $(foreach p,$(VMEM),-DVM_$(p))

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR)
PARAMS += $(foreach p,$(VMEM),-G$(p))

TOP := foldsim_vmem

include ../common.mk
