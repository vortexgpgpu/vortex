ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dxa_s2g_bulk_groups_ws
SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

override CONFIGS += -DVX_CFG_EXT_DXA_ENABLE \
                    -DVX_CFG_EXT_DXA_GROUP_ENABLE \
                    -DVX_CFG_EXT_DXA_S2G_ENABLE \
                    -DVX_CFG_NUM_WARPS=8 \
                    -DVX_CFG_NUM_DXA_CORES=2 \
                    -DVX_CFG_DXA_GROUP_CONTEXTS=64

# Functional-model-only stress target. The knob delays the first source
# request of each high-level S2G transfer; it is intentionally not part of the
# RTL configuration or ISA. Keep it separate so ordinary regression remains a
# zero-latency correctness test.
S2G_STRESS_LATENCY ?= 20000
S2G_STRESS_CONFIGS := $(CONFIGS) \
                      -DVX_CFG_DXA_S2G_DEBUG_LATENCY=$(S2G_STRESS_LATENCY)

SRCS := $(SRC_DIR)/main.cpp
VX_SRCS := $(SRC_DIR)/kernel.cpp
KERNEL_LIB := vortex2

include ../common.mk

.PHONY: run-simx-stress
run-simx-stress: $(PROJECT) kernel.vxbin
	CONFIGS="$(S2G_STRESS_CONFIGS)" XLEN="$(XLEN)" \
	  $(MAKE) -C $(VORTEX_RT_SRC)/simx DESTDIR=$(VORTEX_RT_LIB)
	LD_LIBRARY_PATH=$(VORTEX_RT_LIB):$(LD_LIBRARY_PATH) \
	  VORTEX_DRIVER=simx ./$(PROJECT) $(OPTS)
