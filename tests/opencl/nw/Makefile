ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := nw

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/nw.cpp

nw.cl: $(SRC_DIR)/nw.cl
	cp $< $@

KERNEL_SRCS := nw.cl

# OPTS ?= 2048 10 ./nw.cl
OPTS ?= 128 2 ./nw.cl # vortex

include ../common.mk
