ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Eight writes on AXI at once behind the smallest queue, 16-byte rows packed four to a line, an 8-entry table, 6-bit write numbers, write first.
PROJECT := foldsim_vmem_w8q10

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(FOLDSIM_SHELL)/rtl

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem

# The controller's parameters, to the RTL and the test alike.
VMEM := QUEUE=10 OUTSTANDING=4 WRITES_OUT=8 HAZARD_BITS=3 SEQ_W=6 DW=128 MW=16 ROW_BYTES=16 RDW_READ_FIRST=0

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(FOLDSIM_SHELL)/include
CXXFLAGS += $(foreach p,$(VMEM),-DVM_$(p))

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR)
PARAMS += $(foreach p,$(VMEM),-G$(p))

TOP := foldsim_vmem

include ../common.mk
