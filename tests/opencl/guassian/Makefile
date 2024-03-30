ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := guassian

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc $(SRC_DIR)/clutils.cpp $(SRC_DIR)/utils.cpp

# usage: [-f <input_file>] [-s <size>]
OPTS ?= -q -s 32

include ../common.mk
