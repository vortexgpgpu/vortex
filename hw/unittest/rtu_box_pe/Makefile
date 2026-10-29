ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := rtu_box_pe

RTL_DIR := $(VORTEX_HOME)/hw/rtl
SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)
SIMX_DIR := $(VORTEX_HOME)/sim/simx

# The reference is the SimX model itself, so the RTU config must match both sides.
CONFIGS += -DSIMULATION -DVX_CFG_EXT_RTU_ENABLE

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I$(SIMX_DIR) -I$(SIMX_DIR)/rtu -I$(SIM_COMMON_DIR)
CXXFLAGS += -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include

SRCS += $(SIMX_DIR)/rtu/rtu_isect.cpp
SRCS += $(SRC_DIR)/main.cpp

PARAMS := -GTAG_WIDTH=32

RTL_PKGS += $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/fpu/VX_fpu_pkg.sv $(RTL_DIR)/rtu/VX_rtu_pkg.sv
RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces
RTL_INCLUDE += -I$(RTL_DIR)/fpu -I$(RTL_DIR)/rtu -I$(SRC_DIR)

VL_FLAGS += -I$(ROOT_DIR)/hw

TOP := VX_rtu_box_pe_tb

include ../common.mk
