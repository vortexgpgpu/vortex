ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Three controllers of 16 instances x 64 rows on one master with the kernel's 1-bit ids: every id in use by several at once.
PROJECT := foldsim_vmem_share

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem_share

# The controllers' and the sharing's parameters, to the RTL and the test alike.
VMEM := NC=3 IW=4 AW=6 AXI_ID_W=1

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common
CXXFLAGS += $(foreach p,$(VMEM),-DVM_$(p))

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR) -I$(SRC_DIR)
PARAMS += $(foreach p,$(VMEM),-G$(p))

TOP := foldsim_vmem_share_tb

include ../common.mk
