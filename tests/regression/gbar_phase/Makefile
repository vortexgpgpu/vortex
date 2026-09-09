ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The global barrier only exists when a cluster holds more than one core
# (VX_bar_unit's USE_GBAR = VX_CFG_NUM_CORES > 1), so default to 2 cores
# unless the caller pinned a core count already.
CONFIGS := $(if $(findstring -DVX_CFG_NUM_CORES,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_NUM_CORES=2)

PROJECT := gbar_phase

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk
