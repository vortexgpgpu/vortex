ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := gfx_pipeline_kernel

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)
SETUP_DIR := $(VORTEX_HOME)/sw/gfx

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# Reuse the setup front-end headers (setup_math.h / setup_types.h). Bit-exact
# setup bbox vs host Binning(): forbid FP contraction (fmadd) so the device
# evaluates a*b+c with the same rounding as the x86 reference.
VX_CFLAGS += -I$(SETUP_DIR) -ffp-contract=off
CXXFLAGS  += -I$(SETUP_DIR) -ffp-contract=off

OPTS ?= -n200

KERNEL_LIB := vortex2

include ../common.mk
