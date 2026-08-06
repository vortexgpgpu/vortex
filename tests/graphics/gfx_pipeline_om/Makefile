ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Full FF chain: RASTER (coverage) + TEX (compiled, unused here) + OM (ROP).
CONFIGS := $(if $(findstring -DVX_CFG_EXT_RASTER_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RASTER_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_OM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_OM_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TEX_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TEX_ENABLE)
# vx_om4 stages its payload in the shared graphics window (SETW), owned at the
# SFU level (GfxWindow) and available whenever OM / TEX is built — RTU is NOT
# required (the A2 decouple). This config exercises the OM-without-RTU path.

PROJECT := gfx_pipeline_om

SRC_DIR := $(VORTEX_HOME)/tests/graphics/$(PROJECT)
SETUP_DIR := $(VORTEX_HOME)/sw/gfx

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_CFLAGS += -I$(VORTEX_HOME)/sw/common -I$(SETUP_DIR) -I$(VORTEX_HOME)/tests/graphics/common -I$(THIRD_PARTY_DIR) -ffp-contract=off

# Reuse gfx_draw3d's trace + reference assets (don't duplicate the binaries).
CXXFLAGS += -I$(VORTEX_HOME)/sw/common -I$(SETUP_DIR) -I$(VORTEX_HOME)/tests/graphics/common -I$(THIRD_PARTY_DIR) -DASSETS_PATHS='"$(SRC_DIR),$(VORTEX_HOME)/tests/graphics/gfx_draw3d"' -ffp-contract=off

LDFLAGS += $(THIRD_PARTY_DIR)/cocogfx/libcocogfx.a -lpng -lz -lboost_serialization

KERNEL_LIB := vortex2

OPTS ?= -w128 -h128 -r triangle_ref_128.png

include ../common.mk
