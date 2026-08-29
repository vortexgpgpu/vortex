ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Each fixed-function stage can be dropped independently with NO_TEX / NO_OM /
# NO_RASTER (the same knobs tests/vulkan/common.mk exposes). The unit is then
# absent from the build and the kernel routes that stage through its gfx_sw
# software emitter, so any FF/SW mix -- up to all-software -- renders the scene
# against the same golden. Routing is keyed off VX_CFG_EXT_*_ENABLED, never a
# runtime flag: a stage cannot take the FF path on a device that has no FF unit.
ifneq ($(NO_TEX),1)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TEX_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TEX_ENABLE)
endif
ifneq ($(NO_OM),1)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_OM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_OM_ENABLE)
endif
ifneq ($(NO_RASTER),1)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_RASTER_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RASTER_ENABLE)
endif

PROJECT := gfx_draw3d

SRC_DIR := $(VORTEX_HOME)/tests/graphics/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# -ffp-contract=off: keep float codegen FMA-free so the perspective-interp math
# is bit-identical across rv32imaf and rv64imafd (fused a*b+c would round ±1 ULP
# differently between the two targets, flipping texel-boundary pixels). Matches
# the other gfx_pipeline_* / gfx_setup kernels.
VX_CFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR) -ffp-contract=off

# common.h pulls gfx_sw.h for the software TEX/OM emitters; its build contract
# requires the divergence-bbs flag + GFX_SW_DIVERGENCE_OK.
include $(VORTEX_HOME)/sw/gfx/libgfx_sw.mk
VX_CFLAGS += $(LIBGFX_SW_VX_CFLAGS)

CXXFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR) -DASSETS_PATHS='"$(SRC_DIR)"' -ffp-contract=off

LDFLAGS += $(THIRD_PARTY_DIR)/cocogfx/libcocogfx.a -lpng -lz -lboost_serialization

KERNEL_LIB := vortex2

OPTS ?=

include ../common.mk