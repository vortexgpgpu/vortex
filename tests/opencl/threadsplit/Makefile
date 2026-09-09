ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk
PROJECT := threadsplit
SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)
SRCS := $(SRC_DIR)/main.cc
kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@
KERNEL_SRCS := kernel.cl
OPTS ?= --benchmark tl_cg --size 128 --resources 1
include ../common.mk
