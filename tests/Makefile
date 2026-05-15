ROOT_DIR := $(realpath ..)
include $(ROOT_DIR)/config.mk

all: unittest kernel regression opencl riscv

unittest:
	$(MAKE) -C unittest

kernel:
	$(MAKE) -C kernel

regression:
	$(MAKE) -C regression

opencl:
	$(MAKE) -C opencl

riscv:
	$(MAKE) -C riscv

hip:
	$(MAKE) -C hip

clean:
	$(MAKE) -C unittest clean
	$(MAKE) -C kernel clean
	$(MAKE) -C regression clean
	$(MAKE) -C opencl clean
	$(MAKE) -C riscv clean
	$(MAKE) -C hip clean

.PHONY: all unittest kernel regression opencl riscv hip