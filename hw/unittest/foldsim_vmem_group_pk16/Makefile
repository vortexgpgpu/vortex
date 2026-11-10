ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Three members of FireSim's tile, packed: 16-byte rows (four to a line), a 96-bit member, a 64-line cache, known-zero blocks of 4 rows, 32 reads on 2 ids.
PROJECT := foldsim_vmem_group_pk16

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(FOLDSIM_SHELL)/rtl

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_vmem_group

# The setup (foldsim_vmem_group_tb.sv and main.cpp's table), to both sides.
CFG := 5

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -DVG_CFG=$(CFG) -I$(FOLDSIM_SHELL)/include

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR) -I$(SRC_DIR)
CONFIGS += -DVG_CFG_$(CFG)

TOP := foldsim_vmem_group_tb

include ../common.mk
