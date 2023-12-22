all: kernel regression opencl riscv unittest

kernel:
	$(MAKE) -C kernel

regression:
	$(MAKE) -C regression

opencl:
ifneq ($(XLEN),64)
	$(MAKE) -C opencl
endif

riscv:
	$(MAKE) -C riscv

unittest:
	$(MAKE) -C unittest

clean:
	$(MAKE) -C kernel clean
	$(MAKE) -C regression clean
	$(MAKE) -C opencl clean
	$(MAKE) -C riscv clean
	$(MAKE) -C unittest clean

clean-all:
	$(MAKE) -C kernel clean
	$(MAKE) -C regression clean-all
	$(MAKE) -C opencl clean-all
	$(MAKE) -C riscv clean
	$(MAKE) -C unittest clean

.PHONY: all kernel regression opencl riscv unittest