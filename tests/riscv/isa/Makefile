include ../common.mk

TEST_DIR := $(VORTEX_HOME)/tests/riscv/isa

# TODO: Missing features
# - unaligned LD/ST (RTL)
# - 64-bit float extension (RTL)
# - 64-bit double extension (RTL)
# - compressed extension
# - fence extension
# - atomics extension
# - vector extension

TESTS_32I := $(filter-out $(TEST_DIR)/rv32ui-p-ma_data.hex $(TEST_DIR)/rv32ui-p-fence_i.hex, $(wildcard $(TEST_DIR)/rv32ui-p-*.hex))
TESTS_32M := $(wildcard $(TEST_DIR)/rv32um-p-*.hex)
TESTS_32A := $(wildcard $(TEST_DIR)/rv32ua-p-*.hex)
TESTS_32F := $(wildcard $(TEST_DIR)/rv32uf-p-*.hex)
TESTS_32D := $(wildcard $(TEST_DIR)/rv32ud-p-*.hex)

TESTS_64I := $(filter-out $(TEST_DIR)/rv64ui-p-ma_data.hex $(TEST_DIR)/rv64ui-p-fence_i.hex, $(wildcard $(TEST_DIR)/rv64ui-p-*.hex))
TESTS_64M := $(wildcard $(TEST_DIR)/rv64um-p-*.hex)
TESTS_64A := $(wildcard $(TEST_DIR)/rv64ua-p-*.hex)
TESTS_64F := $(wildcard $(TEST_DIR)/rv64uf-p-*.hex)
TESTS_64FX := $(filter-out $(TEST_DIR)/rv64uf-p-fcvt.hex $(TEST_DIR)/rv64uf-p-fcvt_w.hex, $(wildcard $(TEST_DIR)/rv64uf-p-*.hex))
TESTS_64D := $(wildcard $(TEST_DIR)/rv64ud-p-*.hex)

all:

run-simx-32imafd:
	$(foreach test, $(TESTS_32I) $(TESTS_32M) $(TESTS_32F) $(TESTS_32D) $(TESTS_32A), $(SIM_DIR)/simx/simx -r $(test) || exit;)

run-simx-64imafd:
	$(foreach test, $(TESTS_64I) $(TESTS_64M) $(TESTS_64F) $(TESTS_64D) $(TESTS_64A), $(SIM_DIR)/simx/simx -r $(test) || exit;)

run-simx-32i:
	$(foreach test, $(TESTS_32I), $(SIM_DIR)/simx/simx -r $(test) || exit;)

run-simx-32im:
	$(foreach test, $(TESTS_32I) $(TESTS_32M), $(SIM_DIR)/simx/simx -r $(test) || exit;)
	
run-simx-32f:
	$(foreach test, $(TESTS_32F), $(SIM_DIR)/simx/simx -r $(test) || exit;)

run-simx-32d:
	$(foreach test, $(TESTS_32D), $(SIM_DIR)/simx/simx -r $(test) || exit;)

run-simx-64im:
	$(foreach test, $(TESTS_64I) $(TESTS_64M), $(SIM_DIR)/simx/simx -r $(test) || exit;)

run-simx-64f:
	$(foreach test, $(TESTS_64F), $(SIM_DIR)/simx/simx -r $(test) || exit;)

run-simx-64fx:
	$(foreach test, $(TESTS_64FX), $(SIM_DIR)/simx/simx -r $(test) || exit;)

run-simx-64d:
	$(foreach test, $(TESTS_64D), $(SIM_DIR)/simx/simx -r $(test) || exit;)

run-simx-32: run-simx-32imafd

run-simx-64: run-simx-32imafd run-simx-64imafd

run-simx: run-simx-$(XLEN)

run-rtlsim-32imf:
	$(foreach test, $(TESTS_32I) $(TESTS_32M) $(TESTS_32F), $(SIM_DIR)/rtlsim/rtlsim -r $(test) || exit;)

run-rtlsim-64imf:
	$(foreach test, $(TESTS_64I) $(TESTS_64M) $(TESTS_64F), $(SIM_DIR)/rtlsim/rtlsim -r $(test) || exit;)

run-rtlsim-32i:
	$(foreach test, $(TESTS_32I), $(SIM_DIR)/rtlsim/rtlsim -r $(test) || exit;)

run-rtlsim-32im:
	$(foreach test, $(TESTS_32I) $(TESTS_32M), $(SIM_DIR)/rtlsim/rtlsim -r $(test) || exit;)

run-rtlsim-32f:
	$(foreach test, $(TESTS_32F), $(SIM_DIR)/rtlsim/rtlsim -r $(test) || exit;)

run-rtlsim-64im:
	$(foreach test, $(TESTS_64I) $(TESTS_64M), $(SIM_DIR)/rtlsim/rtlsim -r $(test) || exit;)

run-rtlsim-64f:
	$(foreach test, $(TESTS_64F), $(SIM_DIR)/rtlsim/rtlsim -r $(test) || exit;)

run-rtlsim-64fx:
	$(foreach test, $(TESTS_64FX), $(SIM_DIR)/rtlsim/rtlsim -r $(test) || exit;)

run-rtlsim-64d:
	$(foreach test, $(TESTS_64D), $(SIM_DIR)/rtlsim/rtlsim -r $(test) || exit;)

run-rtlsim-32: run-rtlsim-32imf

run-rtlsim-64: run-rtlsim-32imf run-rtlsim-64imf

run-rtlsim: run-rtlsim-$(XLEN)

clean: