ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := alu_muldiv

RTL_DIR := $(VORTEX_HOME)/hw/rtl

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

# Eight ALU lanes. Without SV_DPI the unit builds its synthesizable multiplier
# and serial divider, the ones simulation otherwise replaces.
CONFIGS += -DVX_CFG_NUM_THREADS=8

RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv

RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(SRC_DIR)

TOP := VX_alu_muldiv_top

include ../common.mk
