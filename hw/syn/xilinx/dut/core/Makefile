PROJECT = VX_core_top
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv
FPU_IP = 1

include ../../common.mk

#CONFIGS += -DVX_CFG_NUM_WARPS=32
#CONFIGS += -DVX_CFG_NUM_THREADS=32

FPU_INCLUDE = -I$(RTL_DIR)/fpu
ifneq (,$(filter -DVX_CFG_FPU_TYPE_FPNEW, $(XCONFIGS)))
	FPU_INCLUDE += -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -J$(THIRD_PARTY_DIR)/cvfpu/src/fpu_div_sqrt_mvp/hdl -J$(THIRD_PARTY_DIR)/cvfpu/src
endif
RTL_INCLUDE = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache $(FPU_INCLUDE)
RTL_INCLUDE += -I$(UNITTEST_DIR)/core

# Per-extension RTL sources (TCU/DXA/V/RTU/TEX/RASTER/OM) —
# single source of truth, appended based on the resolved XCONFIGS.
include $(VORTEX_HOME)/hw/syn/extensions.mk
