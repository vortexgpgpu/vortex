PROJECT = vortex_afu
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv
FPU_IP = 1

include ../../common.mk

#CONFIGS += -DNUM_CORES=2
#CONFIGS += -DNUM_WARPS=32
#CONFIGS += -DNUM_THREADS=32
#CONFIGS += -DL2_ENABLE

FPU_INCLUDE = -I$(RTL_DIR)/fpu
ifneq (,$(findstring FPU_FPNEW,$(CONFIGS)))
	FPU_INCLUDE += -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -J$(THIRD_PARTY_DIR)/cvfpu/src/fpu_div_sqrt_mvp/hdl -J$(THIRD_PARTY_DIR)/cvfpu/src
endif
RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache -I$(AFU_DIR) -I$(AFU_DIR)/ccip $(FPU_INCLUDE)
