ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := rp_sortnet

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_HDRS := $(SRC_DIR)/common.h

# keys each thread sorts in registers (a power of two)
KEYS ?= 64
CXXFLAGS += -DKEYS=$(KEYS)
VX_CFLAGS += -DKEYS=$(KEYS)

OPTS ?= -n1024

KERNEL_LIB := vortex2

include ../common.mk
