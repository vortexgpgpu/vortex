ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := hotspot

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/hotspot.c \
        $(SRC_DIR)/OpenCL_helper_library.c

hotspot_kernel.cl: $(SRC_DIR)/hotspot_kernel.cl
	cp $< $@

KERNEL_SRCS := hotspot_kernel.cl

# OPTS ?= 512 2 2 $(SRC_DIR)/temp_512 $(SRC_DIR)/power_512 output.out
OPTS ?= 64 2 2 $(SRC_DIR)/temp_64 $(SRC_DIR)/power_64 output.out

# CXXFLAGS += -DRD_WG_SIZE=4

include ../common.mk 