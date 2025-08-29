ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := particlefilter

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/ex_particle_OCL_naive_seq.cpp

particle_naive.cl: $(SRC_DIR)/particle_naive.cl
    cp $< $@

KERNEL_SRCS := particle_naive.cl

OPTS ?= -x 128 -y 128 -z 10 -np 10000

include ../common.mk