ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tfu_unit

RTL_DIR := $(VORTEX_HOME)/hw/rtl
SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)

CONFIGS += -DSIMULATION -DVX_CFG_EXT_TFU_ENABLE

SRCS += $(SW_COMMON_DIR)/tfu.cpp
SRCS += $(SRC_DIR)/main.cpp

RTL_PKGS += $(RTL_DIR)/VX_gpu_pkg.sv
RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(RTL_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/tfu

# Generated headers (VX_config.vh, VX_types.vh) live in the build tree; add them
# as an include-search path only, not a source-discovery root.
VL_FLAGS += -I$(ROOT_DIR)/hw

TOP := VX_tfu_pe

include ../common.mk
