ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := srad

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.c \
        $(SRC_DIR)/graphics.c \
        $(SRC_DIR)/resize.c \
        $(SRC_DIR)/timer.c \
        $(SRC_DIR)/opencl.c \
        $(SRC_DIR)/kernel_gpu_opencl_wrapper.c

kernel_gpu_opencl.cl: $(SRC_DIR)/kernel_gpu_opencl.cl
	cp $< $@

main.h: $(SRC_DIR)/main.h
	cp $< $@

image.pgm: $(SRC_DIR)/image.pgm
	cp $< $@

KERNEL_SRCS := kernel_gpu_opencl.cl main.h image.pgm

# OPTS ?= 100 0.5 502 458
OPTS ?= 10 0.2 50 45

include ../common.mk