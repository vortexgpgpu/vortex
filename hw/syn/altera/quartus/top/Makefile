PROJECT = vortex_afu
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

include ../../common.mk

CONFIGS += -set "NOPAE"

CONFIGS += -set "NUM_CORES=4"

CONFIGS += -set "L2_ENABLE"

#CONFIGS += -set "L1_DISABLE" 
#CONFIGS += -set "SM_DISABLE"

FPU_INCLUDE = $(RTL_DIR)/fpu_unit;$(THIRD_PARTY_DIR)/fpnew/src;$(THIRD_PARTY_DIR)/fpnew/src/fpu_div_sqrt_mvp/hdl;$(THIRD_PARTY_DIR)/fpnew/src/common_cells/include;$(THIRD_PARTY_DIR)/fpnew/src/common_cells/src
TEX_INCLUDE = $(RTL_DIR)/tex_unit
RTL_INCLUDE = $(RTL_DIR);$(RTL_DIR)/libs;$(RTL_DIR)/interfaces;$(RTL_DIR)/cache;$(AFU_DIR);$(AFU_DIR)/ccip;$(IP_DIR);$(FPU_INCLUDE);$(TEX_INCLUDE)
