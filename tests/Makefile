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
	$(MAKE) clean -C runtime
	$(MAKE) clean -C regression
	$(MAKE) clean -C opencl
	$(MAKE) clean -C riscv
	$(MAKE) clean -C unittest

.PHONY: all runtime regression opencl riscv unittest