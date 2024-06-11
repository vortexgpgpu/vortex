ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := nearn

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc $(SRC_DIR)/clutils.cpp $(SRC_DIR)/utils.cpp

filelist.log:
	echo "$(SRC_DIR)/cane4_0.db\n$(SRC_DIR)/cane4_1.db" > filelist.log

setup: filelist.log
USE_SETUP := yes

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl filelist.log

OPTS ?= filelist.log

include ../common.mk
