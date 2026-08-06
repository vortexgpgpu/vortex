ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# Directory is tests/opencl/btree; the '+' is avoided in the binary name.
PROJECT := btree

SRC_DIR := $(VORTEX_HOME)/tests/opencl/btree

SRCS := $(SRC_DIR)/main.cc

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

OPTS ?= -n 256 -k 32 -r 16 -s 8

include ../common.mk
