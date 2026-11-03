ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tlb_l2

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/vm/VX_tlb_pkg.sv


RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs -I$(VORTEX_HOME)/hw/unittest/tlb_l2
RTL_INCLUDE += -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm

TOP := VX_tlb_l2_top

include ../common.mk