ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# 66-bit rows in 16 bytes, four to a line, write-first, a 64-line cache, known-zero blocks of 8 rows; 64 reads on 2 ids.
PROJECT := foldsim_vmem_rb16c64

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(FOLDSIM_SHELL)/rtl

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem

# The controller's parameters, to the RTL and the test alike.
VMEM := DW=66 MW=1 RDW_READ_FIRST=0 ROW_BYTES=16 CACHE_ROWS=64 KZ_SHIFT=3 AXI_ID_W=1 OUTSTANDING=64

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(FOLDSIM_SHELL)/include
CXXFLAGS += $(foreach p,$(VMEM),-DVM_$(p))

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR)
PARAMS += $(foreach p,$(VMEM),-G$(p))

TOP := foldsim_vmem

include ../common.mk
