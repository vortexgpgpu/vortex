PROJECT = VX_tcu_unit_top
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv
FPU_IP = 1

CONFIGS += -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_F_DISABLE

include ../../common.mk

RTL_INCLUDE = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem
RTL_INCLUDE += -I$(UNITTEST_DIR)/tcu_unit

# Add TCU extension sources
RTL_INCLUDE += -I$(RTL_DIR)/tcu
ifneq (,$(filter -DVX_CFG_TCU_TYPE_DPI, $(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/dpi
endif
ifneq (,$(filter -DVX_CFG_TCU_TYPE_DSP, $(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/dsp
endif
ifneq (,$(filter -DVX_CFG_TCU_TYPE_BHF, $(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/bhf
RTL_INCLUDE += -J$(THIRD_PARTY_DIR)/hardfloat/source/RISCV
RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/hardfloat/source
endif
ifneq (,$(filter -DVX_CFG_TCU_TYPE_FPNEW, $(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/fpnew
RTL_PKGS += $(THIRD_PARTY_DIR)/cvfpu/src/fpnew_pkg.sv $(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src/cf_math_pkg.sv
RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -I$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -I$(THIRD_PARTY_DIR)/cvfpu/src
endif
ifneq (,$(filter -DVX_CFG_TCU_TYPE_TFR, $(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/tfr
endif
