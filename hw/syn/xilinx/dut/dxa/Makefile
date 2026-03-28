PROJECT = VX_dxa_core_top
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

CONFIGS += -DEXT_DXA_ENABLE

include ../../common.mk

RTL_INCLUDE  = -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces
RTL_INCLUDE += -I$(RTL_DIR)/mem -I$(RTL_DIR)/dxa
RTL_PKGS    += $(RTL_DIR)/dxa/VX_dxa_pkg.sv
# Top-level wrapper (shared with the dxa_core unit test)
RTL_INCLUDE += -I$(VORTEX_HOME)/hw/unittest/dxa_core
