ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dogfood

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

HDRS := $(SRC_DIR)/testcases.h $(SRC_DIR)/common.h
VX_HDRS := $(SRC_DIR)/common.h

OPTS ?= -n64 -xbar -xgbar

KERNEL_LIB := vortex2

include ../common.mk