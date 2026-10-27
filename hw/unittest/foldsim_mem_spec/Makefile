ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# hw/unittest/foldsim_mem's test, on the endpoint with staged token ends that
# takes R's ready a cycle late and offers R and AR's ready for either fire.
PROJECT := foldsim_mem

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_mem

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SIM_COMMON_DIR) -I$(SW_COMMON_DIR)

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR)
PARAMS += -GSTAGED=1 -GSPEC=1
CXXFLAGS += -DFOLDSIM_MEM_SPEC=1

# The token modules are not named after their file.
VL_FLAGS += $(FOLDSIM_RTL_DIR)/foldsim_token.sv

TOP := foldsim_mem

include ../common.mk
