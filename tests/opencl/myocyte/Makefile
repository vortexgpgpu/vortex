ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := myocyte

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc

# kernel.cl is self-contained (model math inlined); the host golden shares the
# same math via myocyte_model.h at host-compile time. No atomics are required.
kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

# Tiny by default: 1 cell, 1 ms integration interval (single adaptive step).
OPTS ?= -t 1 -w 1

include ../common.mk
