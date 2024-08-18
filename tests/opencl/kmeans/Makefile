ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := kmeans

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc $(SRC_DIR)/read_input.c $(SRC_DIR)/rmse.c $(SRC_DIR)/kmeans_clustering.c $(SRC_DIR)/cluster.c $(SRC_DIR)/getopt.c

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

# usage: -f<features> -p<points>
OPTS ?= -f100 -p100

include ../common.mk
