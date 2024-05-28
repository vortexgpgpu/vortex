ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

all:

run-simx:
	$(MAKE) -C isa run-simx
	$(MAKE) -C benchmarks_${XLEN} run-simx

run-rtlsim:
	$(MAKE) -C isa run-rtlsim
	$(MAKE) -C benchmarks_32 run-rtlsim
	$(MAKE) -C benchmarks_${XLEN} run-rtlsim

clean:
