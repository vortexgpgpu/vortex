ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := saxpy

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

OPTS ?= -n1024

include ../common.mk
