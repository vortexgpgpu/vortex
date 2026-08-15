PROJECT = VX_local_mem_top
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

include ../../common.mk

# hw/rtl/cache is on the path for VX_amo_alu: the LMEM banks perform atomics
# with the same combinational ALU the cache uses rather than a second copy.
RTL_INCLUDE = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache
RTL_INCLUDE += -I$(UNITTEST_DIR)/local_mem