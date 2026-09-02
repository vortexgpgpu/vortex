ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dxa_group_tracker

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw
CXXFLAGS += -I$(VORTEX_HOME)/sim/simx

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

CONFIGS += -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_EXT_DXA_GROUP_ENABLE

RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/dxa/VX_dxa_pkg.sv $(RTL_DIR)/VX_trace_pkg.sv

RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(DPI_DIR) \
               -I$(RTL_DIR)/libs -I$(RTL_DIR)/dxa \
               -I$(VORTEX_HOME)/hw/unittest/$(PROJECT)

TOP := VX_dxa_group_tracker_top

include ../common.mk
