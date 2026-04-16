ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := yolov3tiny_sparse

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

# Host driver: softfloat needed by util.h / vortex runtime infrastructure
SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp

# Device kernel: sparse WMMA instructions + float/spawn intrinsics
VX_SRCS := $(SRC_DIR)/kernel.cpp

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

OPTS ?=

include ../common.mk
