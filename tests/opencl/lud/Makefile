ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := lud

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/lud.cpp $(SRC_DIR)/common.c

CXXFLAGS += -I$(SRC_DIR)

common.h: $(SRC_DIR)/common.h
	cp $< $@

lud_kernel.cl: $(SRC_DIR)/lud_kernel.cl
	cp $< $@

KERNEL_SRCS := lud_kernel.cl common.h

# Usage: ./lud [-s matrix_size] [-v] [-i input_file]
# OPTS ?= -s 1024 -v
OPTS ?= -s 128

include ../common.mk
