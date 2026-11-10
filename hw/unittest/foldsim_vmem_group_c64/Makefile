ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The same members, write-first, with a 64-line host cache the members share.
PROJECT := foldsim_vmem_group_c64

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(FOLDSIM_SHELL)/rtl

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem_group

# The setup (foldsim_vmem_group_tb.sv and main.cpp's table), to both sides.
CFG := 1

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -DVG_CFG=$(CFG) -I$(FOLDSIM_SHELL)/include

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR) -I$(SRC_DIR)
CONFIGS += -DVG_CFG_$(CFG)

TOP := foldsim_vmem_group_tb

include ../common.mk
