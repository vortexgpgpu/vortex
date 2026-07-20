ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := llama2

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

# vortex2 = the KMU kernel runtime, providing _start and __vx_cta_entry.
# Required for multi-entry .vxbin (one entry per transformer op).
KERNEL_LIB := vortex2

# expf/sqrtf/powf/cosf/sinf in the RoPE, softmax and SwiGLU kernels.
VX_LIBS += -lm

# Model + tokenizer live in third_party/llama2c (fetched, not vendored source).
MODEL_DIR := $(VORTEX_HOME)/third_party/llama2c
OPTS ?= $(MODEL_DIR)/stories15M.bin -z $(MODEL_DIR)/tokenizer.bin -n 32

include ../common.mk
