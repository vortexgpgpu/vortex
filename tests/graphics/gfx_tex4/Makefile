ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# vx_tex4 reads/writes the shared graphics window (SETW/GETW). In SimX those ops
# are provided by the RTU model today, so enable RTU alongside TEX; the RTL
# gfx_window serves both. The host harness + assets are reused from gfx_tex; the
# output (sampled via vx_tex4) is diffed against the gfx_tex point-sample golden,
# so PASS proves vx_tex4 == vx_tex per pixel.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TEX_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TEX_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE -DVX_CFG_RTU_BVH_WIDTH=0)

PROJECT := gfx_tex4

SRC_DIR := $(VORTEX_HOME)/tests/graphics/$(PROJECT)
TEX_DIR := $(VORTEX_HOME)/tests/graphics/gfx_tex

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_CFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR)

CXXFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR) -DASSETS_PATHS='"$(TEX_DIR)"'

LDFLAGS += $(THIRD_PARTY_DIR)/cocogfx/libcocogfx.a -lpng -lz

KERNEL_LIB := vortex2

# Sample via vx_tex4 (point filter) and validate against the gfx_tex g0 golden.
OPTS ?= -i palette64.png -r palette64_ref_g0.png -g 0

include ../common.mk
