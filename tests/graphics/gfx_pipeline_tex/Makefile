ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RASTER_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RASTER_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_OM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_OM_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TEX_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TEX_ENABLE)

PROJECT := gfx_pipeline_tex

SRC_DIR := $(VORTEX_HOME)/tests/graphics/$(PROJECT)
SETUP_DIR := $(VORTEX_HOME)/tests/regression/gfx_setup_kernel

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_CFLAGS += -I$(VORTEX_HOME)/sw/common -I$(SETUP_DIR) -I$(THIRD_PARTY_DIR) -ffp-contract=off

CXXFLAGS += -I$(VORTEX_HOME)/sw/common -I$(SETUP_DIR) -I$(THIRD_PARTY_DIR) -ffp-contract=off

LDFLAGS += $(THIRD_PARTY_DIR)/cocogfx/libcocogfx.a -lpng -lz -lboost_serialization

KERNEL_LIB := vortex2

OPTS ?= -w128 -h128

include ../common.mk
