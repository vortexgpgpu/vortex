all: driver runtime regression opencl riscv

driver:
	$(MAKE) -C driver

runtime:
	$(MAKE) -C runtime

regression:
	$(MAKE) -C regression

opencl:
	$(MAKE) -C opencl

riscv:
	$(MAKE) -C riscv

clean:
	$(MAKE) clean -C driver
	$(MAKE) clean -C runtime
	$(MAKE) clean -C regression
	$(MAKE) clean -C opencl
	$(MAKE) clean -C riscv

.PHONY: all driver runtime regression opencl riscv