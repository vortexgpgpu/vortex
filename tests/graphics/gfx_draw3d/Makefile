ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_TEX_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TEX_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_OM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_OM_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_RASTER_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RASTER_ENABLE)
# vx_om4 / vx_tex4 stage their payload in the shared graphics window (SETW),
# owned at the SFU level (GfxWindow) and available whenever OM / TEX is built, so
# RTU is NOT required (the A2 decouple). Full FF pipeline, no RTU.

PROJECT := gfx_draw3d

SRC_DIR := $(VORTEX_HOME)/tests/graphics/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_CFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR)

CXXFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR) -DASSETS_PATHS='"$(SRC_DIR)"'

LDFLAGS += $(THIRD_PARTY_DIR)/cocogfx/libcocogfx.a -lpng -lz -lboost_serialization

KERNEL_LIB := vortex2

# RASTER dispatch v2 (FWD) opt-in: GFX_FWD=1 builds the push/launch fragment
# path (driver warp + FWD-injected fragment waves) instead of the legacy vx_rast
# poll loop. Off by default so the legacy / rtlsim paths are unchanged.
ifdef GFX_FWD
VX_CFLAGS += -DGFX_FWD
CXXFLAGS  += -DGFX_FWD
endif

OPTS ?=

include ../common.mk