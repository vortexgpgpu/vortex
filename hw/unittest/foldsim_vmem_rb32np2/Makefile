ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# 32-byte rows (byte enables), two to a line, 5 instances (the multiply), a 128-line cache, known-zero blocks of 32 rows; 32 reads on 4 ids.
PROJECT := foldsim_vmem_rb32np2

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(FOLDSIM_SHELL)/rtl

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem

# The controller's parameters, to the RTL and the test alike.
VMEM := DW=256 MW=32 IW=3 INSTANCES=5 AW=7 ROW_BYTES=32 CACHE_ROWS=128 KZ_SHIFT=5 AXI_ID_W=2 OUTSTANDING=32

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(FOLDSIM_SHELL)/include
CXXFLAGS += $(foreach p,$(VMEM),-DVM_$(p))

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR)
PARAMS += $(foreach p,$(VMEM),-G$(p))

TOP := foldsim_vmem

include ../common.mk
