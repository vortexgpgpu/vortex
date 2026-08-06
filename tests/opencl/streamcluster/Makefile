ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := streamcluster

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc

# pgain_kernel uses NO atomics (each work-item writes its own work_mem slice),
# so no CONFIGS=-DVX_CFG_EXT_A_ENABLE is required.

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

OPTS ?= -n 128 -d 32 -l 16

include ../common.mk
