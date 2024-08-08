ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := mem_streamer

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(VORTEX_HOME)/sim/common

SRCS := $(DPI_DIR)/util_dpi.cpp
SRCS += $(SRC_DIR)/memsim.cpp $(SRC_DIR)/ram.cpp

DBG_TRACE_FLAGS :=

RTL_PKGS :=

RTL_INCLUDE := -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs

TOP := VX_mem_scheduler

include ../common.mk