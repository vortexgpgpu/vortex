ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := transpose

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc $(SRC_DIR)/oclUtils.cpp $(SRC_DIR)/shrUtils.cpp $(SRC_DIR)/cmd_arg_reader.cpp $(SRC_DIR)/transpose_gold.cpp

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

OPTS ?= -width=128 -height=128

#CXXFLAGS += -D_DEBUG

include ../common.mk
