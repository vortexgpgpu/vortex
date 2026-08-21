ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tensor_rzr4
SRC_DIR := $(VORTEX_HOME)/tests/unittest/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp
SRCS += $(SW_COMMON_DIR)/rvfloats.cpp
SRCS += $(SW_COMMON_DIR)/softfloat_ext.cpp

CXXFLAGS += -I$(VORTEX_HOME)/sw/runtime/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

include ../common.mk
