ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := rp_epilogue

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# Accumulator columns each thread holds in registers at a time (ROW = one pass).
CHUNK ?= 16
CXXFLAGS += -DCHUNK=$(CHUNK)
VX_CFLAGS += -DCHUNK=$(CHUNK)

OPTS ?= -r256

KERNEL_LIB := vortex2

include ../common.mk
