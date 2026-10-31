ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Four controllers with 64-line host caches over 5 instances (the multiply), queue 8, 1-bit ids.
PROJECT := foldsim_vmem_share_c

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem_share

# The controllers' and the sharing's parameters, to the RTL and the test alike.
VMEM := NC=4 IW=3 INSTANCES=5 AW=6 QUEUE=8 AXI_ID_W=1 CACHE_ROWS=64

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common
CXXFLAGS += $(foreach p,$(VMEM),-DVM_$(p))

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR) -I$(SRC_DIR)
PARAMS += $(foreach p,$(VMEM),-G$(p))

TOP := foldsim_vmem_share_tb

include ../common.mk
