ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := mem_to_axi

RTL_DIR := $(VORTEX_HOME)/hw/rtl

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

# ports in, AXI banks out, AXI id width (6 engages the tag buffer)
NI   ?= 4
NB   ?= 4
TIDW ?= 6

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR) -DNI=$(NI) -DNB=$(NB) -DTIDW=$(TIDW)

SRCS += $(SRC_DIR)/main.cpp

RTL_PKGS :=

RTL_INCLUDE := -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs

TOP := VX_mem_to_axi

PARAMS := -GDATA_WIDTH=64 -GADDR_WIDTH_IN=16 -GADDR_WIDTH_OUT=32 -GTAG_WIDTH_IN=8 -GTAG_WIDTH_OUT=$(TIDW)
PARAMS += -GNUM_PORTS_IN=$(NI) -GNUM_BANKS_OUT=$(NB) -GINTERLEAVE=1 -GREQ_OUT_BUF=2 -GRSP_OUT_BUF=2

CONFIGS += -DSIMULATION

include ../common.mk
