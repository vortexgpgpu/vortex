ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := pathfinder

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/OpenCL.cpp $(SRC_DIR)/timing.cpp 

kernels.cl: $(SRC_DIR)/kernels.cl
	cp $< $@

KERNEL_SRCS := kernels.cl

# OPTS ?=  -c 100000 -r 100 -h 20
OPTS ?=  -c 50 -r 50 -h 20 # vortex


include ../common.mk
