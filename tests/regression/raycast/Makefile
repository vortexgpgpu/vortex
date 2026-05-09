ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := raycast

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/tracer.cpp $(SRC_DIR)/mesh.cpp $(SRC_DIR)/surface.cpp $(SRC_DIR)/bvh.cpp $(SRC_DIR)/scene.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

CXXFLAGS += -DASSETS_PATHS='"$(SRC_DIR)"'

# RTLsim is roughly two orders of magnitude slower than SimX; keep the
# default workload small so make-driven sweeps fit in a 60s budget.
# Override OPTS at the command line for full-resolution runs.
OPTS ?= -w 16 -h 12

KERNEL_LIB := vortex2

include ../common.mk