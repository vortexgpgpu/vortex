PROJECT = no_smem

OPTS ?= -n8

VX_SRCS = kernel.cpp $(VORTEX_KN_PATH)/src/vx_perf.c $(VORTEX_KN_PATH)/src/vx_syscalls.c $(VORTEX_KN_PATH)/src/vx_print.S $(VORTEX_KN_PATH)/src/vx_start.S

SRCS = main.cpp

include ../common.mk

VX_CFLAGS += -DSM_DISABLE
VX_LDFLAGS = -Wl,-Bstatic,--gc-sections,-T,$(VORTEX_KN_PATH)/linker/vx_link$(XLEN).ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR)

VX_CC  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-gcc
VX_CXX = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-g++
VX_DP  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objdump
VX_CP  = $(RISCV_TOOLCHAIN_PATH)/bin/$(RISCV_PREFIX)-objcopy