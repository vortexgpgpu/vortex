XLEN ?=32
VLEN ?=256
TOOLDIR ?= /opt

RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv$(XLEN)-gnu-toolchain

all:

run-simx:
	RISCV_TOOLCHAIN_PATH=$(RISCV_TOOLCHAIN_PATH) XLEN=$(XLEN) VLEN=$(VLEN) ./run-test.sh