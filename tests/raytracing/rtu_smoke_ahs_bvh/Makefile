ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)
# CW-BVH4 scene -> build the RTU as a CW-BVH4 walker (exercises the BVH
# walker's per-triangle any-hit path, unlike the flat-walker rtu_smoke_ahs).
CONFIGS += -DVX_CFG_RTU_BVH_WIDTH=4

PROJECT := rtu_smoke_ahs_bvh

SRC_DIR := $(VORTEX_HOME)/tests/raytracing/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# No -d: run BOTH AHS branches in one invocation — IGNORE (candidate ignored ->
# MISS) then ACCEPT (candidate committed -> HIT at t=5). Both require the CW-BVH
# walker to yield an any-hit callback for the non-opaque triangle; the accept
# branch additionally exercises the committed-hit commit path (I23 coverage).
OPTS ?= -n1

KERNEL_LIB := vortex2

include ../common.mk
