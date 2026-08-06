ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# vx_tex4 quad mode (hardware LOD) reads/writes the shared graphics window
# (SETW/GETW), provided by the RTU in SimX; enable RTU alongside TEX. Self-check
# kernel (no PNG reference): each pixel's output is XOR(vx_tex4_quad, vx_tex@HW-LOD),
# which the host verifies is all-zero. Minified (scale 0.5) so the derivative LOD
# is non-zero, exercising the derivative -> log2 datapath.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TEX_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TEX_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE -DVX_CFG_RTU_BVH_WIDTH=0)

PROJECT := gfx_tex4q

SRC_DIR := $(VORTEX_HOME)/tests/graphics/$(PROJECT)
TEX_DIR := $(VORTEX_HOME)/tests/graphics/gfx_tex

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_CFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR)

CXXFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR) -DASSETS_PATHS='"$(TEX_DIR)"'

LDFLAGS += $(THIRD_PARTY_DIR)/cocogfx/libcocogfx.a -lpng -lz

KERNEL_LIB := vortex2

OPTS ?= -i palette64.png -s 0.5

include ../common.mk
