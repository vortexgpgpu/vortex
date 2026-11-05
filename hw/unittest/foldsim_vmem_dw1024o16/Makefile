ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# 1024-bit rows (two-beat bursts), 16 reads on 2 ids: one id's bursts in order, the ids' beats interleaved.
PROJECT := foldsim_vmem_dw1024o16

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem

# The controller's parameters, to the RTL and the test alike.
VMEM := DW=1024 MW=128 AXI_ID_W=1 OUTSTANDING=16 CACHE_ROWS=32

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common
CXXFLAGS += $(foreach p,$(VMEM),-DVM_$(p))

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR)
PARAMS += $(foreach p,$(VMEM),-G$(p))

TOP := foldsim_vmem

include ../common.mk
