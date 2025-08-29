ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := myocyte

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.c \
        $(SRC_DIR)/embedded_fehlberg_7_8.c \
        $(SRC_DIR)/file.c \
        $(SRC_DIR)/kernel_fin.c \
        $(SRC_DIR)/kernel_gpu_opencl_wrapper.c \
        $(SRC_DIR)/master.c \
        $(SRC_DIR)/num.c \
        $(SRC_DIR)/opencl.c \
        $(SRC_DIR)/solver.c \
        $(SRC_DIR)/timer.c

kernel_gpu_opencl.cl: $(SRC_DIR)/kernel_gpu_opencl.cl
	cp $< $@

KERNEL_SRCS := kernel_gpu_opencl.cl

OPTS ?= -time 1

include ../common.mk