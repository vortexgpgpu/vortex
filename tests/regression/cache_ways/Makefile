ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The test measures the memory read counter, which the RTL has only with PERF_ENABLE.
CONFIGS := $(if $(findstring -DPERF_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DPERF_ENABLE)

PROJECT := cache_ways

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk
