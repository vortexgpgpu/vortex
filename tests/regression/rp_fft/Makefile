ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := rp_fft

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# Points each thread transforms in registers per pass (FFT_LEN must be a power of it).
RADIX ?= 4
CXXFLAGS += -DRADIX=$(RADIX)
VX_CFLAGS += -DRADIX=$(RADIX)

OPTS ?= -b64

KERNEL_LIB := vortex2

include ../common.mk
