ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := alu_muldiv

RTL_DIR := $(VORTEX_HOME)/hw/rtl
SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)

SRCS += $(SRC_DIR)/main.cpp

# VX_alu_muldiv as the core builds it (SYNTHESIS-side units, no DPI models),
# four lanes per ALU block
CONFIGS += -DSIMULATION -DVX_CFG_NUM_THREADS=4 -DVX_CFG_NUM_ALU_LANES=4

RTL_PKGS += $(RTL_DIR)/VX_gpu_pkg.sv
RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(SRC_DIR)
VL_FLAGS += -I$(ROOT_DIR)/hw

TOP := VX_alu_muldiv_top

include ../common.mk
