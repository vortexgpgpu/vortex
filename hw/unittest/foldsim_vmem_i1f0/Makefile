ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# One instance, no forwarding from the write buffer (FORWARD 0): a read meeting a write in flight waits for its B.
PROJECT := foldsim_vmem_i1f0

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem

# The controller's parameters, to the RTL and the test alike.
VMEM := IW=1 INSTANCES=1 AW=8 FORWARD=0

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common
CXXFLAGS += $(foreach p,$(VMEM),-DVM_$(p))

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR)
PARAMS += $(foreach p,$(VMEM),-G$(p))

TOP := foldsim_vmem

include ../common.mk
