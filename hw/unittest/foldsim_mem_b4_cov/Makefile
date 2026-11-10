ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# hw/unittest/foldsim_mem_b4 with coverage of the endpoint's late ARs, stalls and room waits (FOLDSIM_MEM_COVER, a
# --public-flat-rw build): what a sweep reached, printed per setup.
PROJECT := foldsim_mem

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(FOLDSIM_SHELL)/rtl

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_mem

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SIM_COMMON_DIR) -I$(SW_COMMON_DIR) -I$(FOLDSIM_SHELL)/include

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR)

# The B4 image's endpoint: unstaged tokens, SPEC 2, 4 ports (CH_SHIFT 2), 6-bit target ids.
PARAMS += -GSTAGED=0 -GSPEC=2 -GCH_SHIFT=2 -GID_W=6
CXXFLAGS += -DFOLDSIM_MEM_SPEC=2 -DFOLDSIM_MEM_CH_SHIFT=2 -DFOLDSIM_MEM_ID_W=6

# The token modules are not named after their file.
VL_FLAGS += $(FOLDSIM_RTL_DIR)/foldsim_token.sv

VL_FLAGS += --public-flat-rw
CXXFLAGS += -DFOLDSIM_MEM_COVER

TOP := foldsim_mem

include ../common.mk
