ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := cache_mem_bp

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

# the L2 of the A100-class profile: 8 requesters with L1-line words, 192 MSHRs per bank;
# reads the memory takes in flight per port (the bank model's read slots)
RD_CAP ?= 32
MSHR   ?= 192
NREQS  ?= 8
WORD   ?= 64
WB     ?= 1

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -DRD_CAP=$(RD_CAP) -DNREQS=$(NREQS) -DWORD_BYTES=$(WORD)

SRCS += $(SRC_DIR)/main.cpp

RTL_PKGS := $(RTL_DIR)/VX_gpu_pkg.sv

RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs
RTL_INCLUDE += -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/cache -I$(VORTEX_HOME)/hw/unittest/cache

TOP := VX_cache_top

# a write-back last-level cache shaped like the L2 (sectored lines, one memory
# port per bank), small enough that most fills evict a dirty line
PARAMS := -GNUM_REQS=$(NREQS) -GMEM_PORTS=4 -GNUM_BANKS=4 -GCACHE_SIZE=16384 -GLINE_SIZE=128 -GSECTOR_SIZE=64
PARAMS += -GNUM_WAYS=4 -GWORD_SIZE=$(WORD) -GMSHR_SIZE=$(MSHR) -GMRSQ_SIZE=4 -GMREQ_SIZE=0 -GLATENCY=4
PARAMS += -GWRITEBACK=$(WB) -GDIRTY_BYTES=0 -GLAZY_FETCH=0 -GAMO_ENABLE=0 -GIS_LLC=1 -GTAG_WIDTH=16

CONFIGS += -DSIMULATION

include ../common.mk
