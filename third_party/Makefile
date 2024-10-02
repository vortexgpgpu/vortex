XLEN ?=32
TOOLDIR ?= /opt

ifeq ($(XLEN),64)
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv64-gnu-toolchain
else
RISCV_TOOLCHAIN_PATH ?= $(TOOLDIR)/riscv-gnu-toolchain
endif

all: fpnew softfloat ramulator

fpnew:

softfloat:
	SPECIALIZE_TYPE=RISCV SOFTFLOAT_OPTS="-fPIC -DSOFTFLOAT_ROUND_ODD -DINLINE_LEVEL=5 -DSOFTFLOAT_FAST_DIV32TO16 -DSOFTFLOAT_FAST_DIV64TO32" $(MAKE) -C softfloat/build/Linux-x86_64-GCC

ramulator/libramulator.so:
	cd ramulator && mkdir -p build && cd build && cmake .. && make -j4
ramulator: ramulator/libramulator.so

riscv-vector-tests:
	PATH=${RISCV_TOOLCHAIN_PATH}/bin:$$PATH $(MAKE) -C riscv-vector-tests -e MARCH=rv${XLEN}imfv XLEN=${XLEN} RISCV_PREFIX=riscv${XLEN}-unknown-elf-

riscv-vector-tests:
	PATH=${RISCV_TOOLCHAIN_PATH}/bin:$$PATH $(MAKE) -C riscv-vector-tests -e MARCH=rv${XLEN}imfv XLEN=${XLEN} RISCV_PREFIX=riscv${XLEN}-unknown-elf-

clean:
	$(MAKE) -C softfloat/build/Linux-x86_64-GCC clean
	rm -rf ramulator/build ramulator/libramulator.so
	$(MAKE) -C riscv-vector-tests clean

.PHONY: all fpnew softfloat ramulator riscv-vector-tests