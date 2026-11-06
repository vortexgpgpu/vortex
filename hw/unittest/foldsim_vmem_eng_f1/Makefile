ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# A page engine's controller (INSTANCES 1, 1-bit AXI id, 128 reads and writes out, no known-zero rows), forwarding.
PROJECT := foldsim_vmem_eng_f1

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem

# The controller's parameters, to the RTL and the test alike.
VMEM := IW=1 INSTANCES=1 AW=12 KNOWN_ZERO=0 AXI_ID_W=1 OUTSTANDING=128

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common
CXXFLAGS += $(foreach p,$(VMEM),-DVM_$(p))

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR)
PARAMS += $(foreach p,$(VMEM),-G$(p))

TOP := foldsim_vmem

include ../common.mk
