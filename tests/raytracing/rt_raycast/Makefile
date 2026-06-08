ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RTU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RTU_ENABLE)
# CW-BVH4 scene -> build the RTU as a CW-BVH4 walker.
CONFIGS += -DVX_CFG_RTU_BVH_WIDTH=4

PROJECT := rt_raycast

SRC_DIR := $(VORTEX_HOME)/tests/raytracing/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# Reuse the regression/raycast OBJ assets (sphere.obj by default).
CXXFLAGS += -DASSETS_PATHS='"$(VORTEX_HOME)/tests/regression/raycast/assets"'

# Default to a small mesh + resolution so the CW-BVH4 fits the RTU
# per-lane pre-fetch budget and the simx run stays within the test cap.
OPTS ?= -m $(VORTEX_HOME)/tests/regression/raycast/assets/cube.obj -w 32 -h 24

KERNEL_LIB := vortex2

include ../common.mk
