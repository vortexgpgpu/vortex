ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := gfx_setup_kernel

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# Shared front-end kernel sources now live in sw/gfx (single source of truth,
# consumed by the pipeline tests and the vortexpipe driver alike).
VX_CFLAGS += -I$(VORTEX_HOME)/sw/gfx -I$(VORTEX_HOME)/sw/common -I$(VORTEX_HOME)/tests/graphics/common
CXXFLAGS  += -I$(VORTEX_HOME)/sw/gfx -I$(VORTEX_HOME)/sw/common -I$(VORTEX_HOME)/tests/graphics/common

# Bit-exact setup parity with host Binning(): forbid FP contraction (fmadd)
# so the device evaluates a*b+c with the same rounding as the x86 reference.
VX_CFLAGS += -ffp-contract=off
CXXFLAGS  += -ffp-contract=off

# The device kernel and the host reference are built from the same shared
# headers, and the test compares their outputs byte for byte. Both directories
# are globbed rather than naming the headers actually included: an edit to any
# of them changes the primitive layout the two sides must agree on, and missing
# one leaves a silently stale build whose mismatch describes the previous layout.
SHARED_HDRS := $(wildcard $(VORTEX_HOME)/sw/gfx/*.h) $(wildcard $(VORTEX_HOME)/sw/common/*.h)
VX_HDRS := $(SHARED_HDRS)
HDRS    := $(SHARED_HDRS)

OPTS ?= -n200

KERNEL_LIB := vortex2

include ../common.mk
