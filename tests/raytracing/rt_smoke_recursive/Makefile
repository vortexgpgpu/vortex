ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)
# Flat triangle-list scene -> build the RTU as a flat walker (true-HW model).
CONFIGS += -DVX_CFG_RTU_BVH_WIDTH=0
# The sub-ray is traced from inside the closest-hit callback, so the warp holds
# its parent's slot while it asks for a second one. Two traces in flight per warp
# is two slots; the default pool (one per core) cannot serve it and the warp
# would block on a slot only its own CONTINUE could release. The proper fix is to
# retire the parent slot when the record is written, before the closest-hit
# shader runs at all -- what DXR and Vulkan both specify -- which needs the
# handle to stop being the slot index. Until then the pool is sized for the
# nesting.
CONFIGS += -DVX_CFG_RTU_NUM_SLOTS=2

PROJECT := rt_smoke_recursive

SRC_DIR := $(VORTEX_HOME)/tests/raytracing/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?=

KERNEL_LIB := vortex2

include ../common.mk
