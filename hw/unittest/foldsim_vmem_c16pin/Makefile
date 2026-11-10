ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# 16 lines over 256 instances: every line shared by 16 instances, read-first pairs pin theirs.
PROJECT := foldsim_vmem_c16pin

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(FOLDSIM_SHELL)/rtl

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem

# The controller's parameters, to the RTL and the test alike.
VMEM := CACHE_ROWS=16

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(FOLDSIM_SHELL)/include
CXXFLAGS += $(foreach p,$(VMEM),-DVM_$(p))

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR)
PARAMS += $(foreach p,$(VMEM),-G$(p))

TOP := foldsim_vmem

include ../common.mk
