ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# SWEEP_LSU_BASELINE=1 builds the per-thread-load path (no DXA define).
ifeq ($(SWEEP_LSU_BASELINE),)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)
endif

PROJECT := dxa_copy_persist

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

KERNEL_LIB := vortex2

OPTS ?= -d1 -s0 65536 -t0 256 -g 8

include ../common.mk
