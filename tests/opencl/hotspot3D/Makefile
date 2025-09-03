ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := hotspot3D

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/3D.c \
        $(SRC_DIR)/CL_helper.c

hotspotKernel.cl: $(SRC_DIR)/hotspotKernel.cl
	cp $< $@

KERNEL_SRCS := hotspotKernel.cl

#OPTS ?= 512 8 100 $(SRC_DIR)/power_512x8 $(SRC_DIR)/temp_512x8 output.out
# OPTS ?= 64 8 100 $(SRC_DIR)/power_64x8 $(SRC_DIR)/temp_64x8 output.out
OPTS ?= 64 8 1 $(SRC_DIR)/power_64x8 $(SRC_DIR)/temp_64x8 output.out

# CXXFLAGS += -DRD_WG_SIZE=4

include ../common.mk 