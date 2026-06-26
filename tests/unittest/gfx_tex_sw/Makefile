ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := gfx_tex_sw

SRC_DIR := $(VORTEX_HOME)/tests/unittest/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/gfx_render.cpp

# FF model (gfx_render.cpp) pulls in cocogfx (third_party), the generated
# VX_types.h (build sw/), and vx_print.h (sw/kernel/include).
CXXFLAGS += -I$(THIRD_PARTY_DIR) -I$(ROOT_DIR)/sw -I$(VORTEX_HOME)/sw/kernel/include

include ../common.mk
