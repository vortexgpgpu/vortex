ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := llama2

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

# rvfloats provides rv_ftoh_s, used to narrow the weights to fp16 on the host.
SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# vortex2 = KMU kernel runtime; required for multi-entry .vxbin.
KERNEL_LIB := vortex2

# WGMMA needs the tensor core. NT=32/IW=4 matches the geometry the matmul
# kernel and the host launch bounds assume (xtileM=16, xtileN=2*WGMMA_NRC).
CONFIGS := $(if $(findstring -DVX_CFG_EXT_TCU_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_TCU_ENABLE)
CONFIGS := $(if $(findstring -DVX_CFG_NUM_THREADS=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_NUM_THREADS=32)
CONFIGS := $(if $(findstring -DVX_CFG_NUM_WARPS=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_NUM_WARPS=8)
CONFIGS := $(if $(findstring -DVX_CFG_ISSUE_WIDTH=,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_ISSUE_WIDTH=4)
CONFIGS += -DVX_CFG_TCU_WGMMA_ENABLE

# WGMMA_NRC sets the CTA tile width: xtileN = 2*WGMMA_NRC. 8 -> 16, matching
# LLAMA_BATCH so a single N-tile covers the batch.
WGMMA_NRC ?= 8
LLAMA_BATCH ?= 16
CONFIGS += -DWGMMA_NRC=$(WGMMA_NRC) -DLLAMA_BATCH=$(LLAMA_BATCH)

# expf/sqrtf/powf/cosf/sinf in RoPE, softmax and SwiGLU.
VX_LIBS += -lm

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
LDFLAGS  += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

MODEL_DIR := $(VORTEX_HOME)/third_party/llama2c
OPTS ?= $(MODEL_DIR)/stories260K.bin -z $(MODEL_DIR)/tok512.bin -n 32

include ../common.mk
