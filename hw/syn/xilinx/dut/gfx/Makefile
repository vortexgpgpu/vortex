PROJECT = VX_gfx_top
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv
FPU_IP = 1

# Graphics sandbox default configuration:
#   - 1 socket x 2 cores (real tex/rtu socket arbiters)
#   - all four engines on; early-Z on (raster <-> ocache coherence path)
#   - DXA enabled so the socket memory arbiter carries its full production
#     input set {icache, dcache, tcache, rtcache, DXA gmem}
#   - default engine counts and cache geometry (tcache 8K, rtcache 8K,
#     rcache 4K, ocache 16K)
#   - 1x L2 cache, 256KB (drop -DVX_CFG_L2_ENABLE => PASSTHRU wires)
CONFIGS += -DVX_CFG_EXT_TEX_ENABLE
CONFIGS += -DVX_CFG_EXT_RTU_ENABLE
CONFIGS += -DVX_CFG_EXT_RASTER_ENABLE
CONFIGS += -DVX_CFG_EXT_OM_ENABLE
CONFIGS += -DVX_CFG_EXT_DXA_ENABLE
CONFIGS += -DVX_CFG_RASTER_EARLYZ_ENABLE
CONFIGS += -DVX_CFG_NUM_THREADS=16
CONFIGS += -DVX_CFG_NUM_WARPS=16
CONFIGS += -DVX_CFG_NUM_CORES=2
CONFIGS += -DVX_CFG_SOCKET_SIZE=2
CONFIGS += -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=1
CONFIGS += -DVX_CFG_L2_ENABLE
CONFIGS += -DVX_CFG_L2_SIZE=262144

include ../../common.mk

RTL_INCLUDE  = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces
RTL_INCLUDE += -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache -I$(RTL_DIR)/vm -I$(RTL_DIR)/fpu
RTL_INCLUDE += -I$(RTL_DIR)/tex -I$(RTL_DIR)/rtu -I$(RTL_DIR)/raster -I$(RTL_DIR)/om
RTL_INCLUDE += -I$(RTL_DIR)/dxa
RTL_INCLUDE += -I$(UNITTEST_DIR)/gfx

RTL_PKGS += $(RTL_DIR)/fpu/VX_fpu_pkg.sv
RTL_PKGS += $(RTL_DIR)/dxa/VX_dxa_pkg.sv
RTL_PKGS += $(RTL_DIR)/tex/VX_tex_pkg.sv
RTL_PKGS += $(RTL_DIR)/rtu/VX_rtu_pkg.sv
RTL_PKGS += $(RTL_DIR)/raster/VX_raster_pkg.sv
RTL_PKGS += $(RTL_DIR)/om/VX_om_pkg.sv
