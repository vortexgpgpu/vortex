ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

SIM_DIR := $(VORTEX_HOME)/sim
HW_DIR := $(VORTEX_HOME)/hw

INC_DIR := $(VORTEX_HOME)/sw/runtime/include
RT_COMMON_DIR := $(VORTEX_HOME)/sw/runtime/common

# Project the resolved hardware config as -DVX_CFG_* flags (all backends),
# so runtime code need not #include <VX_config.h>.
CXXFLAGS += $(XCONFIGS)