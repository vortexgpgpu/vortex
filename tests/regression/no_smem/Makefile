PROJECT = no_smem

OPTS ?= -n8

VX_SRCS = kernel.c $(VORTEX_KN_PATH)/src/vx_perf.c $(VORTEX_KN_PATH)/src/vx_syscalls.c $(VORTEX_KN_PATH)/src/vx_print.S $(VORTEX_KN_PATH)/src/vx_start.S

SRCS = main.cpp

include ../common.mk

VX_CFLAGS += -DSM_DISABLE
VX_LDFLAGS = -Wl,-Bstatic,--gc-sections,-T,$(VORTEX_KN_PATH)/linker/vx_link$(XLEN).ld

VX_CC  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-gcc
VX_CXX = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-g++
VX_DP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objdump
VX_CP  = $(RISCV_TOOLCHAIN_PATH)/bin/riscv32-unknown-elf-objcopy