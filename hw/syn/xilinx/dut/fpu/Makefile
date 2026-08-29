# On Xilinx, VX_fma_unit/VX_fdiv_unit/VX_fsqrt_unit select the vendor FP operator IP
# (xil_fma/fdiv/sqrt) for the F32 flush-to-zero path, so the IP must be generated.
FPU_IP = 1

include ../../common.mk

PROJECT = VX_fpu_dsp

ifneq (,$(filter -DVX_CFG_FPU_TYPE_FPNEW, $(XCONFIGS)))
PROJECT = VX_fpu_fpnew
endif
ifneq (,$(filter -DVX_CFG_FPU_TYPE_STD, $(XCONFIGS)))
PROJECT = VX_fpu_std
endif

TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

FPU_INCLUDE = -I$(RTL_DIR)/fpu
ifneq (,$(filter -DVX_CFG_FPU_TYPE_FPNEW, $(XCONFIGS)))
	FPU_INCLUDE += -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -J$(THIRD_PARTY_DIR)/cvfpu/src/fpu_div_sqrt_mvp/hdl -J$(THIRD_PARTY_DIR)/cvfpu/src
endif
RTL_INCLUDE = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces $(FPU_INCLUDE)
