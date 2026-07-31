PROJECT = VX_vm_top
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

# MMU-network sandbox default configuration:
#   - 4x 64B parallel memory requests  (NUM_THREADS=16 => DCACHE_NUM_REQS=4, 64B line)
#   - AMO enabled
#   - 2x L1 MMU (dTLB+iTLB), 8KB-class TLB (L2_TLB_SIZE=512), 4 lanes
#   - 2x L1 cache (dcache+icache), 16KB each, 4 lanes
#   - 1x L2 cache, 256KB, fed by {L1 socket port, PTW} => 2 lanes
#     (PLATFORM_MEMORY_NUM_BANKS=1 => L1_MEM_PORTS=1 => L2_NUM_REQS=2)
#   - 1 PTW with 2 walkers
CONFIGS += -DVX_CFG_VM_ENABLE
CONFIGS += -DVX_CFG_EXT_A_ENABLE
CONFIGS += -DVX_CFG_NUM_THREADS=16
CONFIGS += -DVX_CFG_NUM_WARPS=16
CONFIGS += -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=1
CONFIGS += -DVX_CFG_L2_ENABLE
CONFIGS += -DVX_CFG_L2_SIZE=262144

include ../../common.mk

RTL_INCLUDE  = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces
RTL_INCLUDE += -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache -I$(RTL_DIR)/vm
RTL_INCLUDE += -I$(UNITTEST_DIR)/vm

RTL_PKGS    += $(RTL_DIR)/vm/VX_tlb_pkg.sv
