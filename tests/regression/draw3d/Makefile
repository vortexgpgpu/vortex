ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := draw3d

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(VORTEX_HOME)/sim/common/gfxutil.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp $(VORTEX_HOME)/sim/common/graphics.cpp

VX_CFLAGS += -I$(VORTEX_HOME)/sim/common -I$(THIRD_PARTY_DIR)

CXXFLAGS += -I$(VORTEX_HOME)/sim/common -I$(THIRD_PARTY_DIR) -DASSETS_PATHS='"$(SRC_DIR)"'

LDFLAGS += $(THIRD_PARTY_DIR)/cocogfx/libcocogfx.a -lpng -lz -lboost_serialization

OPTS ?=

include ../common.mk