ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := lsu_agu

RTL_DIR := $(VORTEX_HOME)/hw/rtl
SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)

SRCS += $(SRC_DIR)/main.cpp

# the stack interleave on, at the given machine size (CONFIGS adds to it)
CONFIGS += -DSIMULATION -DVX_CFG_LSU_STACK_INTERLEAVE_ENABLE

RTL_PKGS += $(RTL_DIR)/VX_gpu_pkg.sv
RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/core -I$(SRC_DIR)
VL_FLAGS += -I$(ROOT_DIR)/hw

TOP := VX_lsu_agu_top

include ../common.mk
