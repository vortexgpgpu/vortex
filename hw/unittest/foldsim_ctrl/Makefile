ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# The control endpoint at the register map's limits: 32 memory ports, 96
# classes; a simulator without idle slots or virtual memories.
PROJECT := foldsim_ctrl

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim

SRC_DIR := $(VORTEX_HOME)/hw/unittest/$(PROJECT)

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SW_COMMON_DIR)
CXXFLAGS += -DMEM_PORTS=32 -DNUM_CLASSES=96 -DBUBBLES=0 -DVMEM_BYTES=0 -DVMEM_TEL_N=5

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR)

# The token modules are not named after their file.
VL_FLAGS += $(FOLDSIM_RTL_DIR)/foldsim_token.sv
PARAMS += -GMEM_PORTS=32 -GNUM_CLASSES=96 -GBUBBLES=0 -GVMEM_BYTES=64\'d0 -GVMEM_TEL_N=5

TOP := foldsim_ctrl

include ../common.mk
