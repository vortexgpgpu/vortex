ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := gfx_msaa

SRC_DIR := $(VORTEX_HOME)/tests/unittest/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

# gfx_sw.h pulls in cocogfx (third_party) and the generated VX_types.h (build sw/).
CXXFLAGS += -I$(THIRD_PARTY_DIR) -I$(ROOT_DIR)/sw

include ../common.mk
