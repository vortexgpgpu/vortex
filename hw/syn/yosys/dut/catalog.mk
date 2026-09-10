# DUT catalog for the Yosys/OpenSTA synthesis flow.
#
# The Yosys flow needs no per-DUT Makefile: hw/syn/yosys/Makefile already
# includes hw/syn/extensions.mk, which derives each extension's RTL_PKGS and
# include paths from the resolved XCONFIGS. A DUT is therefore just three
# values -- top module, extra include, defines -- forwarded to that flow.
#
# Tops are the same hw/unittest/*/VX_*_top.sv wrappers the Xilinx gate
# synthesizes, so both gates measure the same modules and a divergence between
# them is meaningful rather than an artefact of two DUT lists.

DUTS := cache core tcu gfx tex raster om rtu dxa vm tensor vortex

UNITTEST_DIR ?= $(VORTEX_HOME)/hw/unittest

cache_TOP := VX_cache_top
cache_INC := -I$(UNITTEST_DIR)/cache
cache_CFG := -DVX_CFG_EXT_A_ENABLE

core_TOP := VX_core_top
core_INC := -I$(UNITTEST_DIR)/core
core_CFG := -DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_EXT_C_ENABLE -DVX_CFG_EXT_A_ENABLE

tcu_TOP := VX_tcu_unit_top
tcu_INC := -I$(UNITTEST_DIR)/tcu_unit
tcu_CFG := -DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_EXT_TCU_ENABLE \
           -DVX_CFG_TCU_TYPE_TFR -DVX_CFG_TCU_FP16_ENABLE -DVX_CFG_TCU_FP8_ENABLE \
           -DVX_CFG_TCU_INT8_ENABLE -DVX_CFG_TCU_INT4_ENABLE -DVX_CFG_TCU_TF32_ENABLE \
           -DVX_CFG_TCU_SPARSE_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE

gfx_TOP := VX_gfx_top
gfx_INC := -I$(UNITTEST_DIR)/gfx
gfx_CFG := -DVX_CFG_EXT_TEX_ENABLE -DVX_CFG_EXT_RTU_ENABLE -DVX_CFG_EXT_RASTER_ENABLE \
           -DVX_CFG_EXT_OM_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_RASTER_EARLYZ_ENABLE \
           -DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_NUM_CORES=2 \
           -DVX_CFG_SOCKET_SIZE=2 -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=1 \
           -DVX_CFG_L2_ENABLE -DVX_CFG_L2_SIZE=262144

tex_TOP := VX_tex_core_top
tex_INC := -I$(UNITTEST_DIR)/tex_core
tex_CFG := -DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_EXT_TEX_ENABLE

raster_TOP := VX_raster_core_top
raster_INC := -I$(UNITTEST_DIR)/raster_core
raster_CFG := -DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_EXT_RASTER_ENABLE

om_TOP := VX_om_core_top
om_INC := -I$(UNITTEST_DIR)/om_core
om_CFG := -DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_EXT_OM_ENABLE

rtu_TOP := VX_rtu_core_top
rtu_INC := -I$(UNITTEST_DIR)/rtu_core
rtu_CFG := -DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_EXT_RTU_ENABLE

dxa_TOP := VX_dxa_core_top
dxa_INC := -I$(UNITTEST_DIR)/dxa_core
dxa_CFG := -DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_EXT_DXA_ENABLE

vm_TOP := VX_vm_top
vm_INC := -I$(UNITTEST_DIR)/vm
vm_CFG := -DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_NUM_CORES=2 -DVX_CFG_VM_ENABLE

tensor_TOP := VX_tensor_top
tensor_INC := -I$(UNITTEST_DIR)/tensor
tensor_CFG := -DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_EXT_TCU_ENABLE \
              -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_TCU_MX_ENABLE \
              -DVX_CFG_EXT_A_ENABLE -DVX_CFG_NUM_CORES=2 -DVX_CFG_SOCKET_SIZE=2 \
              -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=1 -DVX_CFG_L2_ENABLE -DVX_CFG_L2_SIZE=262144

# The whole GPU. Hours-long; nightly only, never a PR gate.
vortex_TOP := Vortex
vortex_INC :=
vortex_CFG :=
