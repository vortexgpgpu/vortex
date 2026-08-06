PROJECT = VX_raster_core_top
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

CONFIGS += -DVX_CFG_EXT_RASTER_ENABLE

include ../../common.mk

RTL_INCLUDE  = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces
RTL_INCLUDE += -I$(RTL_DIR)/mem -I$(RTL_DIR)/raster -I$(RTL_DIR)/gfx
RTL_INCLUDE += -I$(UNITTEST_DIR)/raster_core

# VX_raster_dispatch stages fragment waves into the gfx register window, so it
# imports VX_gfx_window_pkg — compile that package ahead of VX_raster_pkg.
RTL_PKGS    += $(RTL_DIR)/gfx/VX_gfx_window_pkg.sv $(RTL_DIR)/raster/VX_raster_pkg.sv
