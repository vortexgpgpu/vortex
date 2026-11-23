ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := mmu_park
RTL_DIR := $(VORTEX_HOME)/hw/rtl
SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

CXXFLAGS := -I$(VORTEX_HOME)/hw/unittest/common
SRCS := $(SRC_DIR)/main.cpp
RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/vm/VX_tlb_pkg.sv
RTL_INCLUDE := -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(SRC_DIR)

TOP := VX_mmu_park_top
include ../common.mk
