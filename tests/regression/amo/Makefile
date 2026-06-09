ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := amo

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# Per-hart iters — small enough to run quickly under simx, big enough
# to actually exercise contention.
OPTS ?= -n32

KERNEL_LIB := vortex2

include ../common.mk
