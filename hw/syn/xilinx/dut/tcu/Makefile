PROJECT = VX_tcu_unit_top
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv
FPU_IP = 1

CONFIGS += -DEXT_TCU_ENABLE -DEXT_F_DISABLE

include ../../common.mk

RTL_INCLUDE = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem
RTL_INCLUDE += -I$(UNITTEST_DIR)/tcu_unit

# Add TCU extension sources
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
RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/hardfloat/source
endif
ifneq (,$(filter -DTCU_TYPE_TFR, $(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/tfr
endif