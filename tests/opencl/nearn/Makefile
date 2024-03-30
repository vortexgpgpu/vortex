ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := nearn

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc $(SRC_DIR)/clutils.cpp $(SRC_DIR)/utils.cpp

OPTS ?= $(SRC_DIR)/filelist.txt

include ../common.mk
