ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# hw/unittest/foldsim_mem's test, on the endpoint with staged token ends that
# takes R's ready two cycles late and offers R, AR, AW and W's ready for each pair of fires.
PROJECT := foldsim_mem

RTL_DIR := $(VORTEX_HOME)/hw/rtl
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim

SRC_DIR := $(VORTEX_HOME)/hw/unittest/foldsim_mem

CXXFLAGS := -I$(SRC_DIR) -I$(VORTEX_HOME)/hw/unittest/common -I$(SIM_COMMON_DIR) -I$(SW_COMMON_DIR)

SRCS := $(SRC_DIR)/main.cpp

DBG_TRACE_FLAGS :=

RTL_INCLUDE := -I$(FOLDSIM_RTL_DIR)

# Write-after-read: the B4 image's endpoint (unstaged tokens, SPEC 2, 4 ports, 6-bit ids) with reads and later
# writes of one block outstanding at once, against an HBM that answers a read with the block as its data leaves
# (main.cpp FOLDSIM_MEM_WAR: +war=0.6 +rdepart, tails 5 %): every read must see the writes before it only.
PARAMS += -GSTAGED=0 -GSPEC=2 -GCH_SHIFT=2 -GID_W=6
CXXFLAGS += -DFOLDSIM_MEM_SPEC=2 -DFOLDSIM_MEM_CH_SHIFT=2 -DFOLDSIM_MEM_ID_W=6 -DFOLDSIM_MEM_WAR

# The token modules are not named after their file.
VL_FLAGS += $(FOLDSIM_RTL_DIR)/foldsim_token.sv

TOP := foldsim_mem

include ../common.mk
