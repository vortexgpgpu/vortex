ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := serial_muldiv

RTL_DIR := $(VORTEX_HOME)/hw/rtl

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)

SRCS += $(SRC_DIR)/main.cpp

RTL_PKGS :=

RTL_INCLUDE := -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(SRC_DIR)

TOP := VX_serial_muldiv_top

CONFIGS += -DSIMULATION

include ../common.mk
