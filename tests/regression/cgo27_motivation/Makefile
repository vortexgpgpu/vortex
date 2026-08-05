ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# cgo27_motivation exercises all 5 GEMM paths in one build:
#   in-core SIMT, in-core TCU, in-core TCU + DXA, DTCU, DTCU + DTCU_TMA.
# Needs the in-core TCU, the DTCU, and the DXA engine all enabled.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TCU_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DTCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DTCU_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)
# Atomics: dtensor_check() reads the descriptor completion flag with an AMO, which
# invalidates the reader's stale L1 copy and resolves at the LLC. A plain load would
# cache the flag and spin on it forever.
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)

# --- Experiment HW config (Hopper-proportional; ~1/4 H100 SM per core). ---
# See README.md for the scaling rationale. Override on the make/blackbox line
# to sweep (e.g. CONFIGS="-DVX_CFG_NUM_CORES=8").
CONFIGS += -DVX_CFG_NUM_CLUSTERS=1
CONFIGS += -DVX_CFG_NUM_CORES=4
CONFIGS += -DVX_CFG_SOCKET_SIZE=1
CONFIGS += -DVX_CFG_NUM_WARPS=16
CONFIGS += -DVX_CFG_NUM_THREADS=32
CONFIGS += -DVX_CFG_ISSUE_WIDTH=4
CONFIGS += -DVX_CFG_NUM_BARRIERS=32          # pipelined double-buffer: 2 barriers/CTA
CONFIGS += -DVX_CFG_DCACHE_SIZE=32768        # 32 KB L1
CONFIGS += -DVX_CFG_LMEM_LOG_SIZE=16         # 64 KB shared memory (2^16)
CONFIGS += -DVX_CFG_L2_ENABLE                # enable L2
CONFIGS += -DVX_CFG_L2_CACHE_SIZE=1048576    # 1 MB L2

PROJECT := cgo27_motivation

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

KERNEL_LIB := vortex2

include ../common.mk
