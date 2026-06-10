ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := gfx_setup_kernel

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# Shared front-end kernel sources now live in sw/gfx (single source of truth,
# consumed by the pipeline tests and the vortexpipe driver alike).
VX_CFLAGS += -I$(VORTEX_HOME)/sw/gfx
CXXFLAGS  += -I$(VORTEX_HOME)/sw/gfx

# Bit-exact setup parity with host Binning(): forbid FP contraction (fmadd)
# so the device evaluates a*b+c with the same rounding as the x86 reference.
VX_CFLAGS += -ffp-contract=off
CXXFLAGS  += -ffp-contract=off

OPTS ?= -n200

KERNEL_LIB := vortex2

include ../common.mk
