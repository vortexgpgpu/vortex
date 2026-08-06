ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RASTER_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RASTER_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_OM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_OM_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TEX_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TEX_ENABLE)
# vx_om4 / vx_tex4 stage their payload in the shared graphics window (SETW). That
# window is owned at the SFU level (GfxWindow) and exists whenever OM / TEX is
# built, so RTU is NOT required — this config exercises the OM+TEX-without-RTU
# path that the A2 decouple enables.

PROJECT := gfx_pipeline_tex

SRC_DIR := $(VORTEX_HOME)/tests/graphics/$(PROJECT)
SETUP_DIR := $(VORTEX_HOME)/sw/gfx

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_CFLAGS += -I$(VORTEX_HOME)/sw/common -I$(SETUP_DIR) -I$(VORTEX_HOME)/tests/graphics/common -I$(THIRD_PARTY_DIR) -ffp-contract=off

CXXFLAGS += -I$(VORTEX_HOME)/sw/common -I$(SETUP_DIR) -I$(VORTEX_HOME)/tests/graphics/common -I$(THIRD_PARTY_DIR) -ffp-contract=off

LDFLAGS += $(THIRD_PARTY_DIR)/cocogfx/libcocogfx.a -lpng -lz -lboost_serialization

KERNEL_LIB := vortex2

OPTS ?= -w128 -h128

include ../common.mk
