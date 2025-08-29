ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := cfd

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/euler3d.cpp 

Kernels.cl: $(SRC_DIR)/Kernels.cl
	cp $< $@

KERNEL_SRCS := Kernels.cl

OPTS ?= $(SRC_DIR)/fvcorr.domn.2097k -t gpu -d 0


include ../common.mk
