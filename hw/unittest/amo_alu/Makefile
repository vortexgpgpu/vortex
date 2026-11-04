ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := amo_alu

RTL_DIR := $(VORTEX_HOME)/hw/rtl
SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

DATA_WIDTH  ?= 64
ARITH_WIDTH ?= $(DATA_WIDTH)

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -DDATA_WIDTH=$(DATA_WIDTH) -DARITH_WIDTH=$(ARITH_WIDTH)

SRCS += $(SRC_DIR)/main.cpp

CONFIGS += -DSIMULATION -DVX_CFG_EXT_A_ENABLE -DVX_CFG_EXT_ZACAS_ENABLE

RTL_PKGS += $(RTL_DIR)/VX_gpu_pkg.sv
RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/cache
VL_FLAGS += -I$(ROOT_DIR)/hw

TOP := VX_amo_alu

PARAMS := -GDATA_WIDTH=$(DATA_WIDTH) -GARITH_WIDTH=$(ARITH_WIDTH)

include ../common.mk
