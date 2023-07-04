PROJECT = Vortex
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

include ../../common.mk

CONFIGS += -DEXT_GFX_ENABLE

CONFIGS += -DNUM_CORES=4

#CONFIGS += -DL1_DISABLE
#CONFIGS += -DSM_DISABLE
#CONFIGS += -DRCACHE_DISABLE -DOCACHE_DISABLE -DTCACHE_DISABLE
#CONFIGS += -DL2_ENABLE

FPU_INCLUDE = -I$(RTL_DIR)/fpu -I$(THIRD_PARTY_DIR)/fpnew/src -I$(THIRD_PARTY_DIR)/fpnew/src/fpu_div_sqrt_mvp/hdl -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/include -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/src
GFX_INCLUDE = -I$(RTL_DIR)/tex -I$(RTL_DIR)/raster -I$(RTL_DIR)/rop
RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache -I$(IP_CACHE_DIR) $(FPU_INCLUDE) $(GFX_INCLUDE)
