ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := hybridsort

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/bucketsort.c $(SRC_DIR)/mergesort.c $(SRC_DIR)/hybridsort.c

bucketsort_kernels.cl: $(SRC_DIR)/bucketsort_kernels.cl
	cp $< $@

histogram1024.cl: $(SRC_DIR)/histogram1024.cl
	cp $< $@

mergesort.cl: $(SRC_DIR)/mergesort.cl
	cp $< $@

KERNEL_SRCS := bucketsort_kernels.cl histogram1024.cl mergesort.cl

OPTS ?= r

include ../common.mk 