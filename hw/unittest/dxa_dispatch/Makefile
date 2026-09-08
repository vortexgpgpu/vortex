ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dxa_dispatch
RTL_DIR := $(VORTEX_HOME)/hw/rtl
SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)
CXXFLAGS := -I$(SRC_DIR) -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(SW_COMMON_DIR)
SRCS := $(SRC_DIR)/main.cpp
VL_FLAGS += -DVX_CFG_EXT_DXA_ENABLE
RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/dxa/VX_dxa_pkg.sv
RTL_INCLUDE := -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/dxa -I$(SRC_DIR)
TOP := VX_dxa_dispatch_top

include ../common.mk
