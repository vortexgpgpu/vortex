PROJECT = no_smem

OPTS ?= -n8

VX_CFLAGS += -DSM_DISABLE

VX_LDFLAGS = -Wl,-Bstatic,-T,$(VORTEX_KN_PATH)/linker/vx_link$(XLEN).ld -Wl,--gc-sections

VX_RUNTIME = $(VORTEX_KN_PATH)/src/vx_start.S $(VORTEX_KN_PATH)/src/vx_perf.c $(VORTEX_KN_PATH)/src/vx_syscalls.c

VX_SRCS = kernel.c $(VX_RUNTIME)

SRCS = main.cpp

VX_SRCS = kernel.c

include ../common.mk
