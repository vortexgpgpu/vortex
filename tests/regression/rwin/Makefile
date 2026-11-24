ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

CONFIGS := $(if $(findstring -DVX_CFG_EXT_RWIN_ENABLE,$(CONFIGS)),$(CONFIGS),$(CONFIGS) -DVX_CFG_EXT_RWIN_ENABLE)

PROJECT := rwin

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)

SRCS := $(SRC_DIR)/main.cpp

VX_SRCS := $(SRC_DIR)/kernel.cpp

VX_HDRS := $(SRC_DIR)/common.h

OPTS ?= -n2

KERNEL_LIB := vortex2

include ../common.mk

# The kernel sets windows in inline asm, which the compiler's register count
# cannot see. With the compiler's windows off it records no count, and the
# loader takes the registers from the kernel's vx_rwin instructions.
LLVM_CFLAGS += -mllvm -riscv-vortex-max-regs=32
