PROJECT = VX_vm_top
TOP_LEVEL_ENTITY = $(PROJECT)
SRC_FILE = $(PROJECT).sv

# MMU-network sandbox default configuration:
#   - 2 cores => 2 sockets, each 4x 64B parallel memory requests
#     (NUM_THREADS=16 => DCACHE_NUM_REQS=4, 64B line)
#   - AMO enabled
#   - per socket: 2x L1 MMU (dTLB+iTLB) + 2x L1 cache (dcache+icache, 16KB, 4 lanes)
#   - 1x L2 cache, 256KB, fed by {2 socket L1 ports, PTW} => 3 lanes
#     (PLATFORM_MEMORY_NUM_BANKS=1 => L1_MEM_PORTS=1 => L2_NUM_REQS=3)
#   - shared L2 TLB (L2_TLB_SIZE=512) + 1 PTW with 2 walkers
CONFIGS += -DVX_CFG_VM_ENABLE
CONFIGS += -DVX_CFG_EXT_A_ENABLE
CONFIGS += -DVX_CFG_NUM_CORES=2
CONFIGS += -DVX_CFG_NUM_THREADS=16
CONFIGS += -DVX_CFG_NUM_WARPS=16
CONFIGS += -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=1
CONFIGS += -DVX_CFG_L2_ENABLE
CONFIGS += -DVX_CFG_L2_SIZE=262144
# L1 hit latency 3 (PIPE_EX=1): registers the way-select so the tag->data
# write path is broken. Matches the multicore integration DUT (top).
CONFIGS += -DVX_CFG_DCACHE_LATENCY=3

include ../../common.mk

RTL_INCLUDE  = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces
RTL_INCLUDE += -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache -I$(RTL_DIR)/vm
RTL_INCLUDE += -I$(UNITTEST_DIR)/vm

RTL_PKGS    += $(RTL_DIR)/vm/VX_tlb_pkg.sv
