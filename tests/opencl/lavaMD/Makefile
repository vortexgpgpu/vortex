ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := lavaMD

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.c \
        $(SRC_DIR)/kernel_gpu_opencl_wrapper.c \
        $(SRC_DIR)/num.c \
        $(SRC_DIR)/timer.c \
        $(SRC_DIR)/opencl.c

kernel_gpu_opencl.cl: $(SRC_DIR)/kernel_gpu_opencl.cl
	cp $< $@

KERNEL_SRCS := kernel_gpu_opencl.cl

OPTS ?= -boxes1d 10

include ../common.mk 