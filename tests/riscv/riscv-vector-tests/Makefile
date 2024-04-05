XLEN ?=32
VLEN ?=256
TOOLDIR ?= /opt

ifeq ($(XLEN),64)
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv64-gnu-toolchain
else
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv-gnu-toolchain
endif

all:

run-simx:
	RISCV_TOOLCHAIN_PATH=$(RISCV_TOOLCHAIN_PATH) XLEN=$(XLEN) VLEN=$(VLEN) ./run-test.sh