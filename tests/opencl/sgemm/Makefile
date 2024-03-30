ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := sgemm

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc

CXXFLAGS += -I$(SRC_DIR)
K_CFLAGS += -I$(SRC_DIR)

OPTS ?= -n32

include ../common.mk
