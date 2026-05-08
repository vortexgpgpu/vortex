PROJECT = VX_raster_core_top
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

CONFIGS += -DEXT_RASTER_ENABLE

include ../../common.mk

RTL_INCLUDE  = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces
RTL_INCLUDE += -I$(RTL_DIR)/mem -I$(RTL_DIR)/raster
RTL_INCLUDE += -I$(UNITTEST_DIR)/raster_core

RTL_PKGS    += $(RTL_DIR)/raster/VX_raster_pkg.sv
