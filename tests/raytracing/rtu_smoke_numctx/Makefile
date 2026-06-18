ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)
# Flat triangle-list scene -> build the RTU as a flat walker (true-HW model).
CONFIGS += -DVX_CFG_RTU_BVH_WIDTH=0
# Decouple the in-flight ray-context pool from SIMD width: NUM_CTX = 8 > NUM_LANES
# (= NUM_THREADS = 4). Exercises the NUM_CTX > NUM_LANES ray->context mapping
# (VX_rtu_core.sv): a warp's NUM_LANES rays occupy the low contexts and the extra
# contexts idle. Locks in that this configuration is functionally correct on both
# simx and rtlsim (the production GPU config raises NUM_CTX above thread count).
CONFIGS += -DVX_CFG_RTU_NUM_CTX=8

PROJECT := rtu_smoke_numctx

# Reuse the rtu_smoke sources verbatim — only the RTU context-pool size differs.
SRC_DIR := $(VORTEX_HOME)/tests/raytracing/rtu_smoke

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -n1

KERNEL_LIB := vortex2

include ../common.mk
