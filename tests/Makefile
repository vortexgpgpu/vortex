ROOT_DIR := $(realpath ..)
include $(ROOT_DIR)/config.mk

all: unittest kernel regression opencl opengl riscv

unittest:
	$(MAKE) -C unittest

kernel:
	$(MAKE) -C kernel

regression:
	$(MAKE) -C regression

opencl:
	$(MAKE) -C opencl

opengl:
	$(MAKE) -C opengl

riscv:
	$(MAKE) -C riscv

clean:
	$(MAKE) -C unittest clean
	$(MAKE) -C kernel clean
	$(MAKE) -C regression clean
	$(MAKE) -C opencl clean
	$(MAKE) -C opengl clean
	$(MAKE) -C riscv clean

.PHONY: all unittest kernel regression opencl opengl riscv