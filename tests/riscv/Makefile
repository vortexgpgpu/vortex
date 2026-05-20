include common.mk

# `install` and the $(RISCV_TESTS_STAMP) rule come from common.mk; it
# clones + builds the upstream riscv-tests ISA suites and benchmarks.

all:

run-simx: $(RISCV_TESTS_STAMP)
	$(MAKE) -C isa run-simx
	$(MAKE) -C benchmarks run-simx

run-rtlsim: $(RISCV_TESTS_STAMP)
	$(MAKE) -C isa run-rtlsim
	$(MAKE) -C benchmarks run-rtlsim

clean:

.PHONY: all run-simx run-rtlsim clean
