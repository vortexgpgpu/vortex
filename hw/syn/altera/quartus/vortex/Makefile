PROJECT = Vortex
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

include ../../common.mk

CONFIGS += -DNUM_CORES=2

CONFIGS += -DL2_ENABLE

#CONFIGS += -DL1_DISABLE

#CONFIGS += -DSM_DISABLE

#CONFIGS += -DEXT_F_DISABLE

#CONFIGS += -DNUM_WARPS=2 -DNUM_THREADS=2

FPU_INCLUDE = -I$(RTL_DIR)/fpu -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/include -I$(THIRD_PARTY_DIR)/fpnew/src/common_cells/src -I$(THIRD_PARTY_DIR)/fpnew/src/fpu_div_sqrt_mvp/hdl -I$(THIRD_PARTY_DIR)/fpnew/src
TEX_INCLUDE = -I$(RTL_DIR)/tex
RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/cache -I$(IP_CACHE_DIR) $(FPU_INCLUDE) $(TEX_INCLUDE)
