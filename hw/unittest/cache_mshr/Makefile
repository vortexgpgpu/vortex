ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := cache_mshr

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

# A non-power-of-two size: the MSHR's id is LOG2UP(MSHR_SIZE) bits wide, so
# 192 entries leave ids 192..255 representable but invalid.
MSHR_SIZE ?= 192

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -DMSHR_SIZE=$(MSHR_SIZE)

SRCS += $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv

RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/cache

TOP := VX_cache_mshr

PARAMS := -GMSHR_SIZE=$(MSHR_SIZE) -GLINE_SIZE=64 -GNUM_BANKS=1 -GDATA_WIDTH=16

include ../common.mk
