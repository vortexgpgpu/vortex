ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tcu_agu

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

# 32 TCU lanes over an LSU of LSU_LANES lanes: R = 32 / LSU_LANES batches.
LSU_LANES ?= 32

CONFIGS += -DVX_CFG_NUM_THREADS=32 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_FP16_ENABLE -DVX_CFG_TCU_SPARSE_ENABLE
CONFIGS += -DVX_CFG_NUM_LSU_LANES=$(LSU_LANES)

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -DTCU_LANES=32 -DLSU_LANES=$(LSU_LANES)

SRCS += $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/tcu/VX_tcu_pkg.sv

# MUTANT_DIR: a directory whose RTL files shadow the tree's (mutation checks)
RTL_INCLUDE := $(if $(MUTANT_DIR),-I$(MUTANT_DIR))
RTL_INCLUDE += -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/tcu -I$(SRC_DIR)

TOP := VX_tcu_agu_top

include ../common.mk
