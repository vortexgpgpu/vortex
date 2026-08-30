PROJECT = VX_afu_wrap
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv
FPU_IP = 1

include ../../common.mk

#CONFIGS += -DVX_CFG_NUM_CORES=2
#CONFIGS += -DVX_CFG_NUM_WARPS=32
#CONFIGS += -DVX_CFG_NUM_THREADS=32
#CONFIGS += -DVX_CFG_L2_ENABLE

FPU_INCLUDE = -I$(RTL_DIR)/fpu
ifneq (,$(filter -DVX_CFG_FPU_TYPE_FPNEW, $(XCONFIGS)))
	FPU_INCLUDE += -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -J$(THIRD_PARTY_DIR)/cvfpu/src/fpu_div_sqrt_mvp/hdl -J$(THIRD_PARTY_DIR)/cvfpu/src
endif
# VX_afu_wrap instantiates the Command Processor; its package/interfaces
# (VX_cp_pkg, VX_cp_axi_m_if, VX_cp_gpu_if, ...) are auto-discovered via -I.
RTL_INCLUDE = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/cache -I$(RTL_DIR)/cp -I$(AFU_DIR) -I$(AFU_COMMON_DIR) $(FPU_INCLUDE)

# Per-extension RTL sources (TCU/DXA/V/RTU/TEX/RASTER/OM) —
# single source of truth, appended based on the resolved XCONFIGS.
include $(VORTEX_HOME)/hw/syn/extensions.mk
