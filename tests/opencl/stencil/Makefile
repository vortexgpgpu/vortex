ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := stencil

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc $(SRC_DIR)/args.c $(SRC_DIR)/parboil_opencl.c $(SRC_DIR)/ocl.c $(SRC_DIR)/gpu_info.c $(SRC_DIR)/file.c

CXXFLAGS += -I$(SRC_DIR)

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

# Usage: #nx #ny #nz #iter -i input_file [-o output_file]
OPTS ?= 32 32 8 1 -i $(SRC_DIR)/32x32x8.bin

include ../common.mk
