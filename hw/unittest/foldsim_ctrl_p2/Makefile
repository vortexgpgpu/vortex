ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# hw/unittest/foldsim_ctrl's test with 2 memory ports and 3 classes: the
# windows end where the ports and classes do.
PROJECT := foldsim_ctrl

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_ctrl

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -DMEM_PORTS=2 -DNUM_CLASSES=3

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR)

# The token modules are not named after their file.
VL_FLAGS += $(FOLDSIM_RTL_DIR)/foldsim_token.sv
PARAMS += -GMEM_PORTS=2 -GNUM_CLASSES=3

TOP := foldsim_ctrl

include ../common.mk
