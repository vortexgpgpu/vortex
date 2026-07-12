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

# -ffp-contract=off: keep float codegen FMA-free so the perspective-interp math
# is bit-identical across rv32imaf and rv64imafd (fused a*b+c would round ±1 ULP
# differently between the two targets, flipping texel-boundary pixels). Matches
# the other gfx_pipeline_* / gfx_setup kernels.
VX_CFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR) -ffp-contract=off

CXXFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR) -DASSETS_PATHS='"$(SRC_DIR)"' -ffp-contract=off

LDFLAGS += $(THIRD_PARTY_DIR)/cocogfx/libcocogfx.a -lpng -lz -lboost_serialization

KERNEL_LIB := vortex2

OPTS ?=

include ../common.mk