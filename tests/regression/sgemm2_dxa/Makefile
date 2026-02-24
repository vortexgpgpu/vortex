ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := sgemm2_dxa

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

OPTS ?= -n16 -t4 -c4 -m2

CONFIGS += -DEXT_DXA_ENABLE -DEXT_TXBAR_ENABLE -DEXT_DXA_CLUSTER_LEVEL_ENABLE

include ../common.mk
