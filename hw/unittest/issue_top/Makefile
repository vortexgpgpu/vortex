ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := issue_top

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(VORTEX_HOME)/sim/common
CXXFLAGS += -I$(ROOT_DIR)/hw

SRCS := $(DPI_DIR)/util_dpi.cpp
SRCS += $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS := -DDBG_TRACE_CACHE

RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/core/VX_trace_pkg.sv

RTL_INCLUDE := -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/mem -I$(RTL_DIR)/core

TOP := VX_issue_top

include ../common.mk