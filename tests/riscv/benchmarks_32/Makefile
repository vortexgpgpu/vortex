include ../common.mk

TEST_DIR := $(VORTEX_HOME)/tests/riscv/benchmarks_32

TESTS := $(wildcard $(TEST_DIR)/*.bin)

all:

run-simx:
	@for test in $(TESTS); do $(SIM_DIR)/simx/simx $$test || exit 1; done

run-rtlsim:
	@for test in $(TESTS); do $(SIM_DIR)/rtlsim/rtlsim $$test || exit 1; done

clean: