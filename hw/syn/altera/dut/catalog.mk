# DUT catalog for the Altera/Quartus synthesis flow.
#
# One entry per DUT, replacing the former dut/<name>/Makefile tree. Included by
# build.mk *around* common.mk: _PRJ/_CFG are needed before it (they feed
# XCONFIGS), _INC/_PKG after it (they consume XCONFIGS and the
# RTL_DIR/UNITTEST_DIR/IP_CACHE_DIR it defines), so _INC/_PKG are deferred
# (`=`, not `:=`).
#
# Add a DUT: append its name to DUTS and define its variables. No new directory,
# no dispatcher edit.

DUTS := cache core fpu issue lmem mem_unit scope top unittest vortex

# ---------------------------------------------------------------------------
# shared include fragments
# ---------------------------------------------------------------------------

BASE_INC = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces

FPU_INC = -I$(RTL_DIR)/fpu $(if $(filter -DVX_CFG_FPU_TYPE_FPNEW,$(XCONFIGS)),\
  -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -J$(THIRD_PARTY_DIR)/cvfpu/src/fpu_div_sqrt_mvp/hdl -J$(THIRD_PARTY_DIR)/cvfpu/src)

# TCU/DXA wiring, gated on the extension being enabled. Same content as the
# xilinx flow's; both are candidates for folding into hw/syn/extensions.mk once
# a before/after check covers it.
TCU_INC = $(if $(filter -DVX_CFG_EXT_TCU_ENABLE,$(XCONFIGS)),-I$(RTL_DIR)/tcu \
  $(if $(filter -DVX_CFG_TCU_TYPE_DPI,$(XCONFIGS)),-I$(RTL_DIR)/tcu/dpi) \
  $(if $(filter -DVX_CFG_TCU_TYPE_DSP,$(XCONFIGS)),-I$(RTL_DIR)/tcu/dsp) \
  $(if $(filter -DVX_CFG_TCU_TYPE_BHF,$(XCONFIGS)),-I$(RTL_DIR)/tcu/bhf -J$(THIRD_PARTY_DIR)/hardfloat/source/RISCV -I$(THIRD_PARTY_DIR)/hardfloat/source) \
  $(if $(filter -DVX_CFG_TCU_TYPE_FPNEW,$(XCONFIGS)),-I$(RTL_DIR)/tcu/fpnew -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -J$(THIRD_PARTY_DIR)/cvfpu/src) \
  $(if $(filter -DVX_CFG_TCU_TYPE_TFR,$(XCONFIGS)),-I$(RTL_DIR)/tcu/tfr))

TCU_PKG = $(if $(filter -DVX_CFG_EXT_TCU_ENABLE,$(XCONFIGS)),$(RTL_DIR)/tcu/VX_tcu_pkg.sv \
  $(if $(filter -DVX_CFG_TCU_TYPE_FPNEW,$(XCONFIGS)),$(THIRD_PARTY_DIR)/cvfpu/src/fpnew_pkg.sv $(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src/cf_math_pkg.sv))

DXA_INC = $(if $(filter -DVX_CFG_EXT_DXA_ENABLE,$(XCONFIGS)),-I$(RTL_DIR)/dxa)
DXA_PKG = $(if $(filter -DVX_CFG_EXT_DXA_ENABLE,$(XCONFIGS)),$(RTL_DIR)/dxa/VX_dxa_pkg.sv)

# ---------------------------------------------------------------------------
# per-DUT: _PRJ top module, _CFG defines, _INC includes, _PKG packages
# ---------------------------------------------------------------------------

cache_PRJ := VX_cache_top
cache_INC  = $(BASE_INC) -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/cache -I$(UNITTEST_DIR)/cache

core_PRJ := VX_core_top
core_INC  = $(BASE_INC) -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/cache \
            -I$(IP_CACHE_DIR) $(FPU_INC) $(TCU_INC)
core_PKG  = $(TCU_PKG)

# The only DUT whose top module depends on the configuration.
fpu_PRJ  = $(if $(filter -DVX_CFG_FPU_TYPE_FPNEW,$(XCONFIGS)),VX_fpu_fpnew,$(if $(filter -DVX_CFG_FPU_TYPE_STD,$(XCONFIGS)),VX_fpu_std,VX_fpu_dsp))
fpu_INC  = $(BASE_INC) -I$(IP_CACHE_DIR) $(FPU_INC)

# NOTE: the original issue/Makefile listed $(FPU_INCLUDE) twice; a duplicate -I
# is a no-op, so it is listed once here.
issue_PRJ := VX_issue_top
issue_INC  = $(BASE_INC) -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm $(FPU_INC) \
             -I$(IP_CACHE_DIR) -I$(UNITTEST_DIR)/issue

lmem_PRJ := VX_local_mem_top
lmem_INC  = $(BASE_INC) -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(UNITTEST_DIR)/local_mem

mem_unit_PRJ := VX_mem_unit_top
mem_unit_INC  = $(BASE_INC) -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/core -I$(RTL_DIR)/fpu \
                -I$(UNITTEST_DIR)/mem_unit

# The only DUT that does not want -I$(RTL_DIR)/interfaces.
scope_PRJ := VX_scope_tap
scope_INC  = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs

top_PRJ := vortex_afu
top_CFG := -DNOPAE -DPLATFORM_PROVIDES_LOCAL_MEMORY
top_INC  = $(BASE_INC) -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/cache \
           -I$(AFU_DIR) -I$(AFU_DIR)/ccip -I$(IP_CACHE_DIR) $(FPU_INC) $(TCU_INC) $(DXA_INC)
top_PKG  = $(TCU_PKG) $(DXA_PKG)

unittest_PRJ := Unittest
unittest_INC  = $(BASE_INC) -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/cache \
                -I$(IP_CACHE_DIR) $(FPU_INC)

vortex_PRJ := Vortex
vortex_INC  = $(BASE_INC) -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/cache \
              -I$(IP_CACHE_DIR) $(FPU_INC) $(TCU_INC) $(DXA_INC)
vortex_PKG  = $(TCU_PKG) $(DXA_PKG)
