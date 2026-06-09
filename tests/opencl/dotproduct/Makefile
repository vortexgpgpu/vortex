ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dotproduct

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc $(SRC_DIR)/oclUtils.cpp $(SRC_DIR)/shrUtils.cpp $(SRC_DIR)/cmd_arg_reader.cpp

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

OPTS ?= -size=4096

include ../common.mk
