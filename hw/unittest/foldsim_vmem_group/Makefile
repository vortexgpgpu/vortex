ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Three members of 8 instances (512-bit rows with byte enables, 64-bit, and a
# 26-bit whole-row one) behind one controller, no host cache, read-first.
PROJECT := foldsim_vmem_group

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem_group

# The setup (foldsim_vmem_group_tb.sv and main.cpp's table), to both sides.
CFG := 0

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -DVG_CFG=$(CFG)

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR) -I$(SRC_DIR)
CONFIGS += -DVG_CFG_$(CFG)

TOP := foldsim_vmem_group_tb

include ../common.mk
