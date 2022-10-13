PROJECT = basic

SRCS = main.cpp

VX_SRCS = kernel.c start.S

OPTS ?= -n256

include ../common.mk

VX_LDFLAGS = -Wl,-Bstatic,-T,$(VORTEX_KN_PATH)/linker/vx_link$(XLEN).ld -Wl,--gc-sections
