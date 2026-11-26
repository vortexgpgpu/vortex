ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := rp_bigmul

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_HDRS := $(SRC_DIR)/common.h

# 32-bit limbs per operand
LIMBS ?= 24
CXXFLAGS += -DLIMBS=$(LIMBS)
VX_CFLAGS += -DLIMBS=$(LIMBS)

OPTS ?= -n256 -i4

KERNEL_LIB := vortex2

include ../common.mk
