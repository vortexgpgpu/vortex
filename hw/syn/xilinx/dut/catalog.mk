# DUT catalog for the Xilinx synthesis flow.
#
# One entry per DUT, replacing the former dut/<name>/Makefile tree. Included by
# build.mk *around* common.mk: the _CFG/_PRJ/_IP values are needed before it
# (they feed XCONFIGS), the _INC/_PKG values after it (they consume XCONFIGS and
# the RTL_DIR/UNITTEST_DIR/THIRD_PARTY_DIR it defines). The _INC/_PKG variables
# are therefore deferred (`=`, not `:=`) so their $(filter …$(XCONFIGS)) tests
# evaluate at use, not at parse.
#
# Add a DUT: append its name to DUTS and define the five variables. No new
# directory, no dispatcher edit.

DUTS := cache core cp dxa fpu gfx issue lmem mem_unit om raster rtu \
        scope tcu tensor tex top unittest vm vortex

# ---------------------------------------------------------------------------
# shared include fragments
# ---------------------------------------------------------------------------

BASE_INC = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces

# cvfpu's headers are -J (library) paths, added only for the FPNEW backend.
FPU_INC = -I$(RTL_DIR)/fpu $(if $(filter -DVX_CFG_FPU_TYPE_FPNEW,$(XCONFIGS)),\
  -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -J$(THIRD_PARTY_DIR)/cvfpu/src/fpu_div_sqrt_mvp/hdl -J$(THIRD_PARTY_DIR)/cvfpu/src)

# TCU backend selection. Kept here rather than deferred to extensions.mk so the
# flatten is semantics-preserving; folding these into extensions.mk is a
# separate change with its own before/after check.
TCU_INC = -I$(RTL_DIR)/tcu \
  $(if $(filter -DVX_CFG_TCU_TYPE_DPI,$(XCONFIGS)),-I$(RTL_DIR)/tcu/dpi) \
  $(if $(filter -DVX_CFG_TCU_TYPE_DSP,$(XCONFIGS)),-I$(RTL_DIR)/tcu/dsp) \
  $(if $(filter -DVX_CFG_TCU_TYPE_BHF,$(XCONFIGS)),-I$(RTL_DIR)/tcu/bhf -J$(THIRD_PARTY_DIR)/hardfloat/source/RISCV -I$(THIRD_PARTY_DIR)/hardfloat/source) \
  $(if $(filter -DVX_CFG_TCU_TYPE_FPNEW,$(XCONFIGS)),-I$(RTL_DIR)/tcu/fpnew -I$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -I$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -I$(THIRD_PARTY_DIR)/cvfpu/src) \
  $(if $(filter -DVX_CFG_TCU_TYPE_TFR,$(XCONFIGS)),-I$(RTL_DIR)/tcu/tfr)

TCU_PKG = $(if $(filter -DVX_CFG_TCU_TYPE_FPNEW,$(XCONFIGS)),\
  $(THIRD_PARTY_DIR)/cvfpu/src/fpnew_pkg.sv $(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src/cf_math_pkg.sv)

# ---------------------------------------------------------------------------
# per-DUT: _PRJ top module, _IP FPU IP, _CFG defines, _INC includes, _PKG pkgs,
#          _EXT=1 to include extensions.mk
# ---------------------------------------------------------------------------

cache_PRJ := VX_cache_top
cache_INC  = $(BASE_INC) -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/cache -I$(UNITTEST_DIR)/cache

core_PRJ  := VX_core_top
core_IP   := 1
core_EXT  := 1
core_INC   = $(BASE_INC) -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/cache $(FPU_INC) -I$(UNITTEST_DIR)/core

cp_PRJ := VX_cp_core_top
cp_INC  = $(BASE_INC) -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/cache -I$(RTL_DIR)/fpu -I$(RTL_DIR)/cp -I$(UNITTEST_DIR)/cp_core

dxa_PRJ := VX_dxa_core_top
dxa_CFG := -DVX_CFG_EXT_DXA_ENABLE
dxa_INC  = $(BASE_INC) -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/dxa -I$(UNITTEST_DIR)/dxa_core
dxa_PKG  = $(RTL_DIR)/dxa/VX_dxa_pkg.sv

# The only DUT whose top module depends on the configuration.
fpu_IP  := 1
fpu_PRJ  = $(if $(filter -DVX_CFG_FPU_TYPE_FPNEW,$(XCONFIGS)),VX_fpu_fpnew,$(if $(filter -DVX_CFG_FPU_TYPE_STD,$(XCONFIGS)),VX_fpu_std,VX_fpu_dsp))
fpu_INC  = $(BASE_INC) $(FPU_INC)

gfx_PRJ := VX_gfx_top
gfx_IP  := 1
gfx_CFG := -DVX_CFG_EXT_TEX_ENABLE -DVX_CFG_EXT_RTU_ENABLE -DVX_CFG_EXT_RASTER_ENABLE \
           -DVX_CFG_EXT_OM_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_RASTER_EARLYZ_ENABLE \
           -DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_NUM_CORES=2 \
           -DVX_CFG_SOCKET_SIZE=2 -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=1 \
           -DVX_CFG_L2_ENABLE -DVX_CFG_L2_SIZE=262144
gfx_INC  = $(BASE_INC) -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache -I$(RTL_DIR)/vm -I$(RTL_DIR)/fpu \
           -I$(RTL_DIR)/tex -I$(RTL_DIR)/rtu -I$(RTL_DIR)/raster -I$(RTL_DIR)/om -I$(RTL_DIR)/dxa \
           -I$(UNITTEST_DIR)/gfx
gfx_PKG  = $(RTL_DIR)/fpu/VX_fpu_pkg.sv $(RTL_DIR)/dxa/VX_dxa_pkg.sv $(RTL_DIR)/tex/VX_tex_pkg.sv \
           $(RTL_DIR)/rtu/VX_rtu_pkg.sv $(RTL_DIR)/raster/VX_raster_pkg.sv $(RTL_DIR)/om/VX_om_pkg.sv

# NOTE: the original issue/Makefile listed $(FPU_INCLUDE) twice; a duplicate -I
# is a no-op, so it is listed once here.
issue_PRJ := VX_issue_top
issue_INC  = $(BASE_INC) -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm $(FPU_INC) -I$(UNITTEST_DIR)/issue

lmem_PRJ := VX_local_mem_top
lmem_INC  = $(BASE_INC) -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/cache -I$(UNITTEST_DIR)/local_mem

mem_unit_PRJ := VX_mem_unit_top
mem_unit_INC  = $(BASE_INC) -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/core -I$(RTL_DIR)/fpu -I$(UNITTEST_DIR)/mem_unit

om_PRJ := VX_om_core_top
om_CFG := -DVX_CFG_EXT_OM_ENABLE
om_INC  = $(BASE_INC) -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/om -I$(UNITTEST_DIR)/om_core
om_PKG  = $(RTL_DIR)/om/VX_om_pkg.sv

raster_PRJ := VX_raster_core_top
raster_CFG := -DVX_CFG_EXT_RASTER_ENABLE
raster_INC  = $(BASE_INC) -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/raster -I$(UNITTEST_DIR)/raster_core
raster_PKG  = $(RTL_DIR)/raster/VX_raster_pkg.sv

rtu_PRJ := VX_rtu_core_top
rtu_IP  := 1
rtu_CFG := -DVX_CFG_EXT_RTU_ENABLE
rtu_INC  = $(BASE_INC) -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/fpu -I$(RTL_DIR)/rtu -I$(UNITTEST_DIR)/rtu_core
rtu_PKG  = $(RTL_DIR)/fpu/VX_fpu_pkg.sv $(RTL_DIR)/rtu/VX_rtu_pkg.sv

# The only DUT that does not want -I$(RTL_DIR)/interfaces.
scope_PRJ := VX_scope_tap
scope_INC  = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs

tcu_PRJ := VX_tcu_unit_top
tcu_IP  := 1
tcu_CFG := -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_F_DISABLE
tcu_INC  = $(BASE_INC) -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(UNITTEST_DIR)/tcu_unit $(TCU_INC)
tcu_PKG  = $(TCU_PKG)

tensor_PRJ := VX_tensor_top
tensor_IP  := 1
tensor_CFG := -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE \
              -DVX_CFG_TCU_MX_ENABLE -DVX_CFG_EXT_A_ENABLE \
              -DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_NUM_CORES=2 \
              -DVX_CFG_SOCKET_SIZE=2 -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=1 \
              -DVX_CFG_L2_ENABLE -DVX_CFG_L2_SIZE=262144
tensor_INC  = $(BASE_INC) -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache -I$(RTL_DIR)/vm \
              -I$(RTL_DIR)/fpu -I$(RTL_DIR)/dxa -I$(UNITTEST_DIR)/tensor $(TCU_INC)
tensor_PKG  = $(TCU_PKG) $(RTL_DIR)/tcu/VX_tcu_pkg.sv $(RTL_DIR)/dxa/VX_dxa_pkg.sv

tex_PRJ := VX_tex_core_top
tex_CFG := -DVX_CFG_EXT_TEX_ENABLE
tex_INC  = $(BASE_INC) -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/tex -I$(UNITTEST_DIR)/tex_core
tex_PKG  = $(RTL_DIR)/tex/VX_tex_pkg.sv

top_PRJ := VX_afu_wrap
top_IP  := 1
top_EXT := 1
top_INC  = $(BASE_INC) -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/cache \
           -I$(RTL_DIR)/cp -I$(AFU_DIR) -I$(AFU_COMMON_DIR) $(FPU_INC)

unittest_PRJ := Unittest
unittest_INC  = $(BASE_INC) -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/cache $(FPU_INC)

vm_PRJ := VX_vm_top
vm_CFG := -DVX_CFG_VM_ENABLE -DVX_CFG_EXT_A_ENABLE -DVX_CFG_NUM_CORES=2 \
          -DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 \
          -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=1 -DVX_CFG_L2_ENABLE \
          -DVX_CFG_L2_SIZE=262144 -DVX_CFG_DCACHE_LATENCY=3
vm_INC  = $(BASE_INC) -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache -I$(RTL_DIR)/vm -I$(UNITTEST_DIR)/vm
vm_PKG  = $(RTL_DIR)/vm/VX_tlb_pkg.sv

vortex_PRJ := Vortex
vortex_IP  := 1
vortex_EXT := 1
vortex_INC  = $(BASE_INC) -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/cache $(FPU_INC)
