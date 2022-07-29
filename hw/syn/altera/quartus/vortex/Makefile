PROJECT = Vortex
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

include ../../common.mk

CONFIGS += -set "NUM_CORES=2"

CONFIGS += -set "L2_ENABLE"

#CONFIGS += -set "L1_DISABLE"

#CONFIGS += -set "SM_DISABLE"

#CONFIGS += -set "EXT_F_DISABLE"

#CONFIGS += -set "NUM_WARPS=2" -set "NUM_THREADS=2"

FPU_INCLUDE = $(RTL_DIR)/fpu_unit;$(THIRD_PARTY_DIR)/fpnew/src;$(THIRD_PARTY_DIR)/fpnew/src/fpu_div_sqrt_mvp/hdl;$(THIRD_PARTY_DIR)/fpnew/src/common_cells/include;$(THIRD_PARTY_DIR)/fpnew/src/common_cells/src
TEX_INCLUDE = $(RTL_DIR)/tex_unit
RTL_INCLUDE = $(RTL_DIR);$(RTL_DIR)/libs;$(RTL_DIR)/interfaces;$(RTL_DIR)/cache;$(IP_DIR);$(FPU_INCLUDE);$(TEX_INCLUDE)
