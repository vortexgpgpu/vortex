PROJECT = vortex_afu
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

include ../../common.mk

CONFIGS += -DNOPAE
CONFIGS += -DEXT_GFX_ENABLE

CONFIGS += -DNUM_CORES=16
CONFIGS += -DNUM_CLUSTERS=1

CONFIGS += -DL2_ENABLE

#CONFIGS += -DL1_DISABLE 
#CONFIGS += -DSM_DISABLE
#CONFIGS += -DRCACHE_DISABLE -DOCACHE_DISABLE -DTCACHE_DISABLE

FPU_INCLUDE = -I$(RTL_DIR)/fpu -J$(THIRD_PARTY_DIR)/fpnew/src/common_cells/include -J$(THIRD_PARTY_DIR)/fpnew/src/common_cells/src -J$(THIRD_PARTY_DIR)/fpnew/src/fpu_div_sqrt_mvp/hdl -J$(THIRD_PARTY_DIR)/fpnew/src
TEX_INCLUDE = -I$(RTL_DIR)/tex -I$(RTL_DIR)/raster -I$(RTL_DIR)/rop
RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache -I$(AFU_DIR) -I$(AFU_DIR)/ccip -I$(IP_CACHE_DIR) $(FPU_INCLUDE) $(TEX_INCLUDE)
