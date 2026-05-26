ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RASTER_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RASTER_ENABLE)

PROJECT := gfx_raster

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_CFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR)

CXXFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR) -DASSETS_PATHS='"$(SRC_DIR)"'

LDFLAGS += $(THIRD_PARTY_DIR)/cocogfx/libcocogfx.a -lpng -lz -lboost_serialization

KERNEL_LIB := vortex2

OPTS ?=

include ../common.mk