PROJECT = Vortex
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

include ../../common.mk

CONFIGS += -set "EXT_GFX_ENABLE"

#CONFIGS += -set "L1_DISABLE"

#CONFIGS += -set "SM_DISABLE"

#CONFIGS += -set "RCACHE_DISABLE" -set "OCACHE_DISABLE" -set "TCACHE_DISABLE"

#CONFIGS += -set "EXT_F_DISABLE"

#CONFIGS += -set "NUM_WARPS=2" -set "NUM_THREADS=2"

CONFIGS += -set "NUM_CORES=2"

CONFIGS += -set "L2_ENABLE"

FPU_INCLUDE = -I$(RTL_DIR)/fpu_unit -I$(THIRD_PARTY_DIR)/fpnew/src -I$(THIRD_PARTY_DIR)/fpnew/src/fpu_div_sqrt_mvp/hdl -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/include -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/src
GFX_INCLUDE = -I$(RTL_DIR)/tex_unit -I$(RTL_DIR)/raster_unit -I$(RTL_DIR)/rop_unit
RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/cache -I$(IP_CACHE_DIR) $(FPU_INCLUDE) $(GFX_INCLUDE)
