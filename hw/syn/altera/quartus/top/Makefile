PROJECT = vortex_afu
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

include ../../common.mk

CONFIGS += -DNOPAE

CONFIGS += -DNUM_CORES=16

CONFIGS += -DL2_ENABLE

#CONFIGS += -DL1_DISABLE 
#CONFIGS += -DSM_DISABLE

FPU_INCLUDE = -I$(RTL_DIR)/fpu -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/include -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/src -I$(THIRD_PARTY_DIR)/fpnew/src/fpu_div_sqrt_mvp/hdl -I$(THIRD_PARTY_DIR)/fpnew/src
TEX_INCLUDE = -I$(RTL_DIR)/tex
RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/cache -I$(AFU_DIR) -I$(AFU_DIR)/ccip -I$(IP_CACHE_DIR) $(FPU_INCLUDE) $(TEX_INCLUDE)
