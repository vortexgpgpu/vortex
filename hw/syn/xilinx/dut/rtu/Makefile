PROJECT = VX_rtu_core_top
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

CONFIGS += -DVX_CFG_EXT_RTU_ENABLE

include ../../common.mk

RTL_INCLUDE  = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces
RTL_INCLUDE += -I$(RTL_DIR)/mem -I$(RTL_DIR)/fpu -I$(RTL_DIR)/rtu
RTL_INCLUDE += -I$(UNITTEST_DIR)/rtu_core

RTL_PKGS    += $(RTL_DIR)/fpu/VX_fpu_pkg.sv $(RTL_DIR)/rtu/VX_rtu_pkg.sv
