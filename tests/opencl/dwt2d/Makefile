ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dwt2d

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/components.cpp $(SRC_DIR)/dwt.cpp

com_dwt.cl: $(SRC_DIR)/com_dwt.cl
	cp $< $@

KERNEL_SRCS := com_dwt.cl

OPTS ?= $(SRC_DIR)/192.bmp -d 192x192 -f -5 -l 3

include ../common.mk
