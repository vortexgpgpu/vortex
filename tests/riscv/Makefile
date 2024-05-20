ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

all:

run-simx:
	$(MAKE) -C isa run-simx riscv-vector-tests

run-rtlsim:
	$(MAKE) -C isa run-rtlsim

clean:
