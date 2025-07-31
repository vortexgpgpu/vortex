PROJECT = VX_tcu_top
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

CONFIGS += -DEXT_TCU_ENABLE

include ../../common.mk

RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces

# Add TCU extension sources
RTL_INCLUDE += -I$(RTL_DIR)/tcu
ifneq (,$(findstring -DTCU_DRL, $(CONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/drl
endif
ifneq (,$(findstring -DTCU_BHF, $(CONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/bhf
endif
RTL_INCLUDE += -J$(THIRD_PARTY_DIR)/hardfloat/source/RISCV
RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/hardfloat/source