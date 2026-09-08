ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dxa_s2g_wgmma_epilogue
SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)
WGMMA_NRC ?= 16
S2G_TE_STATS_X ?= 32
S2G_TE_STATS_Y ?= 16
S2G_TE_STATS_Z ?= 2
S2G_TE_STATS_STAGES ?= 3

CONFIGS += -DVX_CFG_EXT_TCU_ENABLE \
           -DVX_CFG_TCU_WGMMA_ENABLE \
           -DWGMMA_RS \
           -DWGMMA_NRC=$(WGMMA_NRC) \
           -DS2G_TE_STATS_X=$(S2G_TE_STATS_X) \
           -DS2G_TE_STATS_Y=$(S2G_TE_STATS_Y) \
           -DS2G_TE_STATS_Z=$(S2G_TE_STATS_Z) \
           -DS2G_TE_STATS_STAGES=$(S2G_TE_STATS_STAGES) \
           -DVX_CFG_EXT_DXA_ENABLE \
           -DVX_CFG_EXT_DXA_GROUP_ENABLE \
           -DVX_CFG_EXT_DXA_S2G_ENABLE \
           -DVX_CFG_L2_ENABLE=1 \
           -DVX_CFG_NUM_THREADS=8 \
           -DVX_CFG_NUM_WARPS=8 \
           -DVX_CFG_ISSUE_WIDTH=2 \
           -DVX_CFG_NUM_DXA_CORES=1

SRCS := $(SRC_DIR)/main.cpp
VX_SRCS := $(SRC_DIR)/kernel.cpp
HDRS := $(SRC_DIR)/common.h
VX_HDRS := $(HDRS)
KERNEL_LIB := vortex2

include ../common.mk


S2G_STRESS_LATENCY ?= 20000
S2G_STRESS_CONFIGS := $(CONFIGS) -DVX_CFG_DXA_S2G_DEBUG_LATENCY=$(S2G_STRESS_LATENCY)

.PHONY: run-simx-stress run-simx-large
run-simx-stress:
	CONFIGS="$(S2G_STRESS_CONFIGS)" XLEN="$(XLEN)" $(MAKE) -B $(PROJECT) kernel.vxbin
	CONFIGS="$(S2G_STRESS_CONFIGS)" XLEN="$(XLEN)" $(MAKE) -C $(VORTEX_RT_SRC)/simx DESTDIR=$(VORTEX_RT_LIB)
	LD_LIBRARY_PATH=$(VORTEX_RT_LIB):$(LD_LIBRARY_PATH) VORTEX_DRIVER=simx ./$(PROJECT) $(OPTS)

# A deeper 3-D statistics tile that still fits the default 16-KiB LMEM.
# WGMMA_NRC stays at 16 because the RS path intentionally accepts only NRC <= 16.
run-simx-large:
	$(MAKE) -B run-simx WGMMA_NRC=16 S2G_TE_STATS_X=24 S2G_TE_STATS_Y=15 S2G_TE_STATS_Z=3 S2G_TE_STATS_STAGES=3
