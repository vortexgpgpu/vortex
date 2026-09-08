ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dxa_tma_attention_s2g
SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

WGMMA_NRC ?= 16
ATT_K_TILES ?= 2
ATT_ITERS ?= 4
ATT_STAGE_GLOBAL_STORE ?= 0
ATT_COMPUTE_WARPS ?= 2
ATT_OUTPUT_STAGES ?= 2

CONFIGS += -DVX_CFG_EXT_TCU_ENABLE \
           -DVX_CFG_TCU_WGMMA_ENABLE \
           -DWGMMA_RS \
           -DWGMMA_NRC=$(WGMMA_NRC) \
           -DATT_K_TILES=$(ATT_K_TILES) \
           -DATT_ITERS=$(ATT_ITERS) \
           -DATT_STAGE_GLOBAL_STORE=$(ATT_STAGE_GLOBAL_STORE) \
           -DATT_COMPUTE_WARPS=$(ATT_COMPUTE_WARPS) \
           -DATT_OUTPUT_STAGES=$(ATT_OUTPUT_STAGES) \
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


.PHONY: run-simx-large
run-simx-large:
	$(MAKE) -B -j1 run-simx ATT_K_TILES=4 ATT_ITERS=8
