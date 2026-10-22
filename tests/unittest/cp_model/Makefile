ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := cp_model

SRC_DIR := $(VORTEX_HOME)/tests/unittest/$(PROJECT)

CXXFLAGS += -I$(VORTEX_HOME)/sim/common
CXXFLAGS += $(shell python3 $(ROOT_DIR)/ci/gen_config.py --config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS) -DVX_CFG_XLEN=$(XLEN)')
SRCS := $(SRC_DIR)/main.cpp $(VORTEX_HOME)/sim/common/cmd_processor.cpp

include ../common.mk
