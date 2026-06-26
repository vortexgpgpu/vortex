ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := gfx_rast_sw

SRC_DIR := $(VORTEX_HOME)/tests/unittest/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

# rast_sw.h pulls in the generated VX_types.h (build sw/) for VX_RASTER_DIM_BITS.
CXXFLAGS += -I$(ROOT_DIR)/sw

include ../common.mk
