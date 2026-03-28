PROJECT = VX_dxa_core_top
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

CONFIGS += -DEXT_DXA_ENABLE

include ../../common.mk

RTL_INCLUDE  = -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces
RTL_INCLUDE += -I$(RTL_DIR)/mem -I$(RTL_DIR)/dxa
RTL_INCLUDE += -I$(UNITTEST_DIR)/dxa_core

RTL_PKGS    += $(RTL_DIR)/dxa/VX_dxa_pkg.sv

