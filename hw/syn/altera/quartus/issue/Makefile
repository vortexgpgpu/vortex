PROJECT = VX_issue_top
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

include ../../common.mk

#CONFIGS += -DNUM_WARPS=32
#CONFIGS += -DNUM_THREADS=32

FPU_INCLUDE = -I$(RTL_DIR)/fpu
ifneq (,$(findstring FPU_FPNEW,$(CONFIGS)))
	FPU_INCLUDE += -J$(THIRD_PARTY_DIR)/fpnew/src/common_cells/include -J$(THIRD_PARTY_DIR)/fpnew/src/common_cells/src -J$(THIRD_PARTY_DIR)/fpnew/src/fpu_div_sqrt_mvp/hdl -J$(THIRD_PARTY_DIR)/fpnew/src
endif
RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem $(FPU_INCLUDE) -I$(IP_CACHE_DIR) $(FPU_INCLUDE)