ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := heartwall

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.c \
        $(SRC_DIR)/kernel_gpu_opencl_wrapper.c \
        $(SRC_DIR)/avilib.c \
        $(SRC_DIR)/avimod.c \
        $(SRC_DIR)/file.c \
        $(SRC_DIR)/timer.c \
        $(SRC_DIR)/opencl.c

main.h: $(SRC_DIR)/main.h
	cp $< $@

kernel_gpu_opencl.cl: $(SRC_DIR)/kernel_gpu_opencl.cl
	cp $< $@

KERNEL_SRCS := kernel_gpu_opencl.cl main.h

OPTS ?= 20

include ../common.mk 