ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Two controllers with 1024-bit rows (two-beat bursts: W in AW's order), 2-bit ids, 16-line host caches.
PROJECT := foldsim_vmem_share_w

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem_share

# The controllers' and the sharing's parameters, to the RTL and the test alike.
VMEM := NC=2 IW=4 AW=6 DW=1024 AXI_ID_W=2 CACHE_ROWS=16

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common
CXXFLAGS += $(foreach p,$(VMEM),-DVM_$(p))

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR) -I$(SRC_DIR)
PARAMS += $(foreach p,$(VMEM),-G$(p))

TOP := foldsim_vmem_share_tb

include ../common.mk
