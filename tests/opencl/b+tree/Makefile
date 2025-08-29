ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := b+tree

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.c $(SRC_DIR)/num.c $(SRC_DIR)/opencl.c $(SRC_DIR)/timer.c $(SRC_DIR)/kernel_gpu_opencl_wrapper.c $(SRC_DIR)/kernel_gpu_opencl_wrapper_2.c

kernel_gpu_opencl.cl: $(SRC_DIR)/kernel_gpu_opencl.cl
	cp $< $@

kernel_gpu_opencl_2.cl: $(SRC_DIR)/kernel_gpu_opencl_2.cl
	cp $< $@

KERNEL_SRCS := kernel_gpu_opencl.cl kernel_gpu_opencl_2.cl

OPTS ?= file $(SRC_DIR)/mil.txt command $(SRC_DIR)/command.txt


include ../common.mk
