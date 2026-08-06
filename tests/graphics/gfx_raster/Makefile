ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RASTER_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RASTER_ENABLE)

PROJECT := gfx_raster

SRC_DIR := $(VORTEX_HOME)/tests/graphics/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_CFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR)

# The -z software fine-rasterizer kernel inlines the recursive gfx_frag_rast.h walk,
# which can exceed the Vortex divergence pass's default 100-BB guard; raise it
# (libgfx_sw.mk also defines GFX_SW_DIVERGENCE_OK, harmless here).
include $(VORTEX_HOME)/sw/gfx/libgfx_sw.mk
VX_CFLAGS += $(LIBGFX_SW_VX_CFLAGS)

CXXFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR) -DASSETS_PATHS='"$(SRC_DIR)"'

LDFLAGS += $(THIRD_PARTY_DIR)/cocogfx/libcocogfx.a -lpng -lz -lboost_serialization

KERNEL_LIB := vortex2

OPTS ?=

include ../common.mk