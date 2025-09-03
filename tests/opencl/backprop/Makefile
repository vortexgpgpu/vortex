ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := backprop

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/backprop_ocl.cpp $(SRC_DIR)/backprop.cpp $(SRC_DIR)/facetrain.cpp $(SRC_DIR)/imagenet.cpp

backprop_kernel.cl: $(SRC_DIR)/backprop_kernel.cl
	cp $< $@

KERNEL_SRCS := backprop_kernel.cl

# OPTS ?= -n 4194304
OPTS ?= -n 4096


include ../common.mk
