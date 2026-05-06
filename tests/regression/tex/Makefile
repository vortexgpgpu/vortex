ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DEXT_TEX_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DEXT_TEX_ENABLE)

PROJECT := tex

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(VORTEX_HOME)/sw/common/gfxutil.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_CFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR)

CXXFLAGS += -I$(VORTEX_HOME)/sw/common -I$(THIRD_PARTY_DIR) -DASSETS_PATHS='"$(SRC_DIR)"'

LDFLAGS += $(THIRD_PARTY_DIR)/cocogfx/libcocogfx.a -lpng -lz

KERNEL_LIB := vortex2

OPTS ?=

include ../common.mk