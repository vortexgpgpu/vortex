PROJECT = Vortex
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv
FPU_IP = 1

include ../../common.mk

#CONFIGS += -DNUM_CORES=2
#CONFIGS += -DNUM_WARPS=32
#CONFIGS += -DNUM_THREADS=32
#CONFIGS += -DL2_ENABLE

CVFPU_INCLUDE = -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -J$(THIRD_PARTY_DIR)/cvfpu/src

FPU_INCLUDE = -I$(RTL_DIR)/fpu
ifneq (,$(filter -DFPU_TYPE_FPNEW, $(XCONFIGS)))
	FPU_INCLUDE += $(CVFPU_INCLUDE) -J$(THIRD_PARTY_DIR)/cvfpu/src/fpu_div_sqrt_mvp/hdl
endif
RTL_INCLUDE = -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache $(FPU_INCLUDE)

# Add TCU extension sources
ifneq (,$(filter -DEXT_TCU_ENABLE, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/tcu/VX_tcu_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/tcu
	ifneq (,$(filter -DTCU_TYPE_DPI, $(XCONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/dpi
	endif
	ifneq (,$(filter -DTCU_TYPE_DSP, $(XCONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/dsp
	endif
	ifneq (,$(filter -DTCU_TYPE_BHF, $(XCONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/bhf
		RTL_INCLUDE += -J$(THIRD_PARTY_DIR)/hardfloat/source/RISCV
		RTL_INCLUDE += -J$(THIRD_PARTY_DIR)/hardfloat/source
	endif
	ifneq (,$(filter -DTCU_TYPE_FPNEW, $(XCONFIGS)))
		RTL_PKGS += $(THIRD_PARTY_DIR)/cvfpu/src/fpnew_pkg.sv $(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src/cf_math_pkg.sv
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/fpnew
		ifeq (,$(filter -DFPU_TYPE_FPNEW, $(XCONFIGS)))
			RTL_INCLUDE += $(CVFPU_INCLUDE)
		endif
	endif
	ifneq (,$(filter -DTCU_TYPE_TFR, $(XCONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/tfr
	endif
endif

# Add DXA extension sources
ifneq (,$(filter -DEXT_DXA_ENABLE, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/dxa/VX_dxa_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/dxa
endif

# Add V extension sources
ifneq (,$(filter -DEXT_V_ENABLE, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/vpu/VX_vpu_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/vpu
endif
