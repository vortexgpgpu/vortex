ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := allocator

RTL_DIR := $(VORTEX_HOME)/hw/rtl

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

SIZE ?= 24

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR) -DSIZE=$(SIZE)

SRCS += $(SRC_DIR)/main.cpp

RTL_PKGS :=

RTL_INCLUDE := -I$(RTL_DIR) -I$(RTL_DIR)/libs

TOP := VX_allocator

PARAMS := -GSIZE=$(SIZE)

CONFIGS += -DSIMULATION

include ../common.mk
