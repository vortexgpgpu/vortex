ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# the wide loads and stores need FLEN 32
ifneq (,$(filter -DVX_CFG_FLEN_32,$(shell python3 $(ROOT_DIR)/ci/gen_config.py --config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS) -DVX_CFG_XLEN=$(XLEN)')))
CONFIGS := $(if $(findstring -DVX_CFG_EXT_LDST_WIDE_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_LDST_WIDE_ENABLE)
endif

PROJECT := ldst_wide

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk
