ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := lbm

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc $(SRC_DIR)/args.c $(SRC_DIR)/parboil_opencl.c $(SRC_DIR)/gpu_info.c $(SRC_DIR)/lbm.c $(SRC_DIR)/ocl.c

CXXFLAGS += -I$(SRC_DIR)

lbm_macros.h: $(SRC_DIR)/lbm_macros.h
	cp $< $@

layout_config.h: $(SRC_DIR)/layout_config.h
	cp $< $@

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl lbm_macros.h layout_config.h

# Usage: #iter [-i input_file] [-o output_file]
OPTS ?= 1 -i $(SRC_DIR)/32_32_8_ldc.of

include ../common.mk
