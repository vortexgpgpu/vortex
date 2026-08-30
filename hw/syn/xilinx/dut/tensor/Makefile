PROJECT = VX_tensor_top
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv
FPU_IP = 1

# Tensor sandbox default configuration:
#   - 1 socket x 2 cores (real DXA request arb + staging-return switch)
#   - TCU: default type, FP16 + WGMMA + MX => all three TCU memory paths
#     (WGMMA LMEM prefetch, TCU_LD metadata loads, ordinary LSU stream)
#   - LMEM enabled, banks = LSU lanes; dcache with AMO
#   - 1x L2 cache, 256KB, fed by {dcache mem ports, DXA gmem} via the
#     socket memory arbiter (drop -DVX_CFG_L2_ENABLE => PASSTHRU wires)
CONFIGS += -DVX_CFG_EXT_TCU_ENABLE
CONFIGS += -DVX_CFG_EXT_DXA_ENABLE
CONFIGS += -DVX_CFG_TCU_WGMMA_ENABLE
CONFIGS += -DVX_CFG_TCU_MX_ENABLE
CONFIGS += -DVX_CFG_EXT_A_ENABLE
CONFIGS += -DVX_CFG_NUM_THREADS=16
CONFIGS += -DVX_CFG_NUM_WARPS=16
CONFIGS += -DVX_CFG_NUM_CORES=2
CONFIGS += -DVX_CFG_SOCKET_SIZE=2
CONFIGS += -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=1
CONFIGS += -DVX_CFG_L2_ENABLE
CONFIGS += -DVX_CFG_L2_SIZE=262144

include ../../common.mk

RTL_INCLUDE  = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces
RTL_INCLUDE += -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache -I$(RTL_DIR)/vm
RTL_INCLUDE += -I$(RTL_DIR)/fpu -I$(RTL_DIR)/dxa
RTL_INCLUDE += -I$(UNITTEST_DIR)/tensor

# Add TCU extension sources
RTL_INCLUDE += -I$(RTL_DIR)/tcu
ifneq (,$(filter -DVX_CFG_TCU_TYPE_DPI, $(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/dpi
endif
ifneq (,$(filter -DVX_CFG_TCU_TYPE_DSP, $(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/dsp
endif
ifneq (,$(filter -DVX_CFG_TCU_TYPE_BHF, $(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/bhf
RTL_INCLUDE += -J$(THIRD_PARTY_DIR)/hardfloat/source/RISCV
RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/hardfloat/source
endif
ifneq (,$(filter -DVX_CFG_TCU_TYPE_FPNEW, $(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/fpnew
RTL_PKGS += $(THIRD_PARTY_DIR)/cvfpu/src/fpnew_pkg.sv $(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src/cf_math_pkg.sv
RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -I$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -I$(THIRD_PARTY_DIR)/cvfpu/src
endif
ifneq (,$(filter -DVX_CFG_TCU_TYPE_TFR, $(XCONFIGS)))
RTL_INCLUDE += -I$(RTL_DIR)/tcu/tfr
endif

RTL_PKGS += $(RTL_DIR)/tcu/VX_tcu_pkg.sv
RTL_PKGS += $(RTL_DIR)/dxa/VX_dxa_pkg.sv
