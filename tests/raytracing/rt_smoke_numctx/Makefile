ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)
# Flat triangle-list scene -> build the RTU as a flat walker (true-HW model).
CONFIGS += -DVX_CFG_RTU_BVH_WIDTH=0
# Two resident traces: RTU_NUM_SLOTS = 2 DERIVES RTU_NUM_CTX = 8 (a slot owns one
# context per thread of the warp it traces). Exercises the multi-slot RTU — two warps
# traversing concurrently, per-slot completion barriers, and the window-write arbiter
# — which the default single-slot build never reaches.
# NUM_CTX is no longer settable on its own: it used to be, and "NUM_CTX > NUM_LANES with
# the extra contexts idle" described a machine where slots did not own their contexts.
CONFIGS += -DVX_CFG_RTU_NUM_SLOTS=2

PROJECT := rt_smoke_numctx

# Reuse the rt_smoke sources verbatim — only the RTU context-pool size differs.
SRC_DIR := $(VORTEX_HOME)/tests/raytracing/rt_smoke

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -n1

KERNEL_LIB := vortex2

include ../common.mk
