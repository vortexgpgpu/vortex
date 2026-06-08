ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RASTER_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RASTER_ENABLE)

PROJECT := gfx_pipeline_raster

SRC_DIR := $(VORTEX_HOME)/tests/graphics/$(PROJECT)
SETUP_DIR := $(VORTEX_HOME)/tests/regression/gfx_setup_kernel

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# Reuse the setup front-end headers; bit-exact setup vs Binning() needs FP
# contraction off.
VX_CFLAGS += -I$(VORTEX_HOME)/sw/common -I$(SETUP_DIR) -I$(THIRD_PARTY_DIR) -ffp-contract=off

# Reuse gfx_raster's trace + reference assets (don't duplicate the binaries).
CXXFLAGS += -I$(VORTEX_HOME)/sw/common -I$(SETUP_DIR) -I$(THIRD_PARTY_DIR) -DASSETS_PATHS='"$(SRC_DIR),$(VORTEX_HOME)/tests/graphics/gfx_raster"' -ffp-contract=off

LDFLAGS += $(THIRD_PARTY_DIR)/cocogfx/libcocogfx.a -lpng -lz -lboost_serialization

KERNEL_LIB := vortex2

OPTS ?= -w128 -h128 -r triangle_ref_128.png

include ../common.mk
