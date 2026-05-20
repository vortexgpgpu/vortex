ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := multikernel

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

# kentry.S is a .S source, so common.mk links the .vxbin directly
# (kernel.cpp + kentry.S + libvortex2.a) — no separate vx_start probe.
VX_SRCS := $(SRC_DIR)/kernel.cpp $(SRC_DIR)/kentry.S

# vortex2 = the KMU kernel runtime, providing _start and __vx_cta_entry.
KERNEL_LIB := vortex2

OPTS ?= -n256

include ../common.mk
