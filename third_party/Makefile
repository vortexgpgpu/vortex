XLEN ?=32
TOOLDIR ?= /opt

ifeq ($(XLEN),64)
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv64-gnu-toolchain
else
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv-gnu-toolchain
endif

all: fpnew softfloat ramulator riscv-vector-tests

fpnew:

softfloat:
	SPECIALIZE_TYPE=RISCV SOFTFLOAT_OPTS="-fPIC -DSOFTFLOAT_ROUND_ODD -DINLINE_LEVEL=5 -DSOFTFLOAT_FAST_DIV32TO16 -DSOFTFLOAT_FAST_DIV64TO32" $(MAKE) -C softfloat/build/Linux-x86_64-GCC

ramulator:
	cd ramulator && git apply ../../miscs/patch/ramulator.patch 2> /dev/null; true
	$(MAKE) -C ramulator libramulator.a

riscv-vector-tests:
	PATH=${RISCV_TOOLCHAIN_PATH}/bin:$$PATH $(MAKE) -C riscv-vector-tests -e MARCH=rv${XLEN}imfv XLEN=${XLEN} RISCV_PREFIX=riscv${XLEN}-unknown-elf-

clean:
	$(MAKE) -C softfloat/build/Linux-x86_64-GCC clean
	$(MAKE) -C ramulator clean
	$(MAKE) -C riscv-vector-tests clean

.PHONY: all fpnew softfloat ramulator riscv-vector-tests