ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The OM exports through its aperture, so this needs nothing but OM.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_OM_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_OM_ENABLE)

PROJECT := gfx_om

SRC_DIR := $(VORTEX_HOME)/tests/graphics/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_CFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR)

# common.h pulls gfx_sw.h for the -S software-OM path; its build contract
# requires the divergence-bbs flag + GFX_SW_DIVERGENCE_OK.
include $(VORTEX_HOME)/sw/gfx/libgfx_sw.mk
VX_CFLAGS += $(LIBGFX_SW_VX_CFLAGS)

CXXFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR) -DASSETS_PATHS='"$(SRC_DIR)"'

LDFLAGS += $(THIRD_PARTY_DIR)/cocogfx/libcocogfx.a -lpng -lz

KERNEL_LIB := vortex2

OPTS ?=

include ../common.mk