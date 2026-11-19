ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := fp8_cvt

SRC_DIR := $(VORTEX_HOME)/tests/unittest/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

include ../common.mk
