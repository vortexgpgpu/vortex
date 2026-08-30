ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := gfx_pipeline_kernel

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)
SETUP_DIR := $(VORTEX_HOME)/sw/gfx

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# Reuse the setup front-end headers (gfx_setup.h / test_setup_dims.h). Bit-exact
# setup bbox vs host Binning(): forbid FP contraction (fmadd) so the device
# evaluates a*b+c with the same rounding as the x86 reference.
VX_CFLAGS += -mllvm -vortex-divergence-max-bbs=65536 -I$(SETUP_DIR) -I$(VORTEX_HOME)/sw/common -I$(VORTEX_HOME)/tests/graphics/common -ffp-contract=off
CXXFLAGS  += -I$(SETUP_DIR) -I$(VORTEX_HOME)/sw/common -I$(VORTEX_HOME)/tests/graphics/common -ffp-contract=off

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
