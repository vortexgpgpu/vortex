ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := llama

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/llama.cpp $(SRC_DIR)/vxmath.cpp

VX_SRCS := $(SRC_DIR)/matmul_kernel.cpp

# Machine configuration. `moti` = the exact machine of tests/regression/cgo27_motivation
# (the microbenchmark that produced vortex_schedule_result(Sheet2).csv); `default` = the
# repository defaults from VX_config.toml (1 core, 4 warps, 4 threads, no L2, no tensor units).
# Same rule as the microbenchmark: never pass CONFIGS= on the make command line, because a
# command-line variable cannot be appended to and every `CONFIGS +=` below is silently dropped.
LLAMA_MACHINE ?= moti

ifeq ($(LLAMA_MACHINE),moti)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TCU_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DTCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DTCU_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_DXA_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_DXA_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_EXT_A_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_A_ENABLE)
CONFIGS += -DVX_CFG_L2_ARB_ENGINE_BYPASS_LIMIT=0
CONFIGS += -DVX_CFG_TCU_WGMMA_ENABLE -DWGMMA_RS -DWGMMA_NRC=8
CONFIGS += -DVX_CFG_NUM_CLUSTERS=1
CONFIGS += -DVX_CFG_NUM_CORES=4
CONFIGS += -DVX_CFG_SOCKET_SIZE=1
CONFIGS := $(if $(findstring -DVX_CFG_NUM_WARPS=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_NUM_WARPS=16)
CONFIGS += -DVX_CFG_NUM_THREADS=32
CONFIGS += -DVX_CFG_ISSUE_WIDTH=4
CONFIGS += -DVX_CFG_NUM_BARRIERS=32
CONFIGS += -DVX_CFG_DCACHE_SIZE=32768
CONFIGS += -DVX_CFG_LMEM_LOG_SIZE=16
CONFIGS += -DVX_CFG_L2_ENABLE
CONFIGS += -DVX_CFG_L2_CACHE_SIZE=1048576
endif

# Extra knobs go here, never in CONFIGS= on the command line.
LLAMA_EXTRA_CONFIGS ?=
CONFIGS += $(LLAMA_EXTRA_CONFIGS)

# LLAMA_CYCLES=1 adds host-only device-cycle accounting (MCYCLE summed over launches,
# printed as one [LLAMA-LEGACY] line). Host flag only, so the kernel and the simulator
# library are unaffected. 0 = the original port's code path.
LLAMA_CYCLES ?= 0
ifeq ($(LLAMA_CYCLES),1)
CXXFLAGS += -DLLAMA_LEGACY_CYCLES
endif

# Deterministic short run by default: fixed prompt, greedy sampling, fixed seed, 8 tokens.
# The original port's default was `-n 40 -v 1` with a time-based seed.
LLAMA_STEPS ?= 8
LLAMA_PROMPT ?= Once upon a time
LLAMA_VORTEX ?= 1
OPTS ?= $(SRC_DIR)/stories15M.bin -z $(SRC_DIR)/tokenizer.bin -i "$(LLAMA_PROMPT)" -t 0 -s 1 -n $(LLAMA_STEPS) -v $(LLAMA_VORTEX)

include ../common.mk

# This directory holds the original llama-c port as written for the 2025 Vortex fork, which
# had no -Werror. Keep its warnings as warnings so the code stays byte-identical to the
# original (sign-compare, designated initializers, string constants to char*, ...).
CXXFLAGS  += -Wno-error
VX_CFLAGS += -Wno-error
