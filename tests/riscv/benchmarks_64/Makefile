include ../common.mk

TEST_DIR := $(VORTEX_HOME)/tests/riscv/benchmarks_64

TESTS := $(wildcard $(TEST_DIR)/*.bin)

all:

run-simx:
	$(foreach test, $(TESTS), $(SIM_DIR)/simx/simx $(test) || exit;)

run-rtlsim:
	$(foreach test, $(TESTS), $(SIM_DIR)/rtlsim/rtlsim $(test) || exit;)

clean: