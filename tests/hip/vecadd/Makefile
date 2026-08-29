ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := vecadd

SRC_DIR := $(VORTEX_HOME)/tests/hip/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

common.h: $(SRC_DIR)/common.h
	cp $< $@

OPTS ?= -n64

include ../common.mk
