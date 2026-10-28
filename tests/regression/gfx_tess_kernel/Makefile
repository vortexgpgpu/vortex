ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := gfx_tess_kernel

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/ref/tessellator.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# The kernels and the host reference both come from the shared sources the
# driver uses (sw/gfx, sw/common); the reference tessellator is in ref/.
VX_CFLAGS += -I$(VORTEX_HOME)/sw/gfx -I$(VORTEX_HOME)/sw/common
CXXFLAGS  += -I$(VORTEX_HOME)/sw/gfx -I$(VORTEX_HOME)/sw/common -I$(SRC_DIR)/ref
# The reference is third-party code, kept as written.
CXXFLAGS  += -Wno-unused-parameter -Wno-unused-variable -Wno-pedantic -Wno-implicit-fallthrough \
             -Wno-sign-compare -Wno-unused-but-set-variable -Wno-misleading-indentation

SHARED_HDRS := $(wildcard $(VORTEX_HOME)/sw/gfx/*.h) $(wildcard $(VORTEX_HOME)/sw/gfx/*.cpp) \
               $(wildcard $(VORTEX_HOME)/sw/common/*.h)
VX_HDRS := $(SHARED_HDRS)
HDRS    := $(SHARED_HDRS) $(SRC_DIR)/common.h

OPTS ?= -n200

KERNEL_LIB := vortex2

include ../common.mk
