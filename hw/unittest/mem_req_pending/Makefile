ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := mem_req_pending

RTL_DIR := $(VORTEX_HOME)/hw/rtl

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

NI ?= 9
NO ?= 1

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR) -DNI=$(NI) -DNO=$(NO)

SRCS += $(SRC_DIR)/main.cpp

RTL_PKGS :=

RTL_INCLUDE := -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/mem

TOP := VX_mem_req_pending

PARAMS := -GNUM_INPUTS=$(NI) -GNUM_OUTPUTS=$(NO)

CONFIGS += -DSIMULATION

include ../common.mk
