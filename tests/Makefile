all: runtime regression opencl riscv unittest

runtime:
	$(MAKE) -C runtime

regression:
	$(MAKE) -C regression

opencl:
	$(MAKE) -C opencl

riscv:
	$(MAKE) -C riscv

unittest:
	$(MAKE) -C unittest

clean:
	$(MAKE) -C runtime clean
	$(MAKE) -C regression clean
	$(MAKE) -C opencl clean
	$(MAKE) -C riscv clean
	$(MAKE) -C unittest clean

clean-all:
	$(MAKE) -C runtime clean
	$(MAKE) -C regression clean-all
	$(MAKE) -C opencl clean-all
	$(MAKE) -C riscv clean
	$(MAKE) -C unittest clean

.PHONY: all runtime regression opencl riscv unittest