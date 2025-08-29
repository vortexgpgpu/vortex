ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := streamcluster

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/streamcluster.cpp

Kernels.cl: $(SRC_DIR)/Kernels.cl
	cp $< $@

KERNEL_SRCS := Kernels.cl

OPTS ?= 10 20 256 65536 65536 1000 none output.txt 1 -t gpu -d 0

include ../common.mk

