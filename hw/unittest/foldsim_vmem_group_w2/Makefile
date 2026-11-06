ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Setup 3 (four members, a 16-line cache, write first) with two writes on AXI at once, a two-entry hazard table and 7-bit write numbers.
PROJECT := foldsim_vmem_group_w2

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem_group

# The setup (foldsim_vmem_group_tb.sv and main.cpp's table), to both sides.
CFG := 3

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -DVG_CFG=$(CFG)

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR) -I$(SRC_DIR)
CONFIGS += -DVG_CFG_$(CFG) -DVG_SMALL_W

TOP := foldsim_vmem_group_tb

include ../common.mk
