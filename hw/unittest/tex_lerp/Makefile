ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tex_lerp

RTL_DIR := $(VORTEX_HOME)/hw/rtl
SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I$(ROOT_DIR)/sw -I$(THIRD_PARTY_DIR)

SRCS := $(SRC_DIR)/main.cpp

RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/tex -I$(SRC_DIR)
VL_FLAGS += -I$(ROOT_DIR)/hw

TOP := VX_tex_lerp_top

include ../common.mk
