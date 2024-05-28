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

TESTS_32I := $(filter-out $(TEST_DIR)/rv32ui-p-ma_data.bin $(TEST_DIR)/rv32ui-p-fence_i.bin, $(wildcard $(TEST_DIR)/rv32ui-p-*.bin))
TESTS_32M := $(wildcard $(TEST_DIR)/rv32um-p-*.bin)
TESTS_32A := $(wildcard $(TEST_DIR)/rv32ua-p-*.bin)
TESTS_32F := $(wildcard $(TEST_DIR)/rv32uf-p-*.bin)
TESTS_32D := $(wildcard $(TEST_DIR)/rv32ud-p-*.bin)

TESTS_64I := $(filter-out $(TEST_DIR)/rv64ui-p-ma_data.bin $(TEST_DIR)/rv64ui-p-fence_i.bin, $(wildcard $(TEST_DIR)/rv64ui-p-*.bin))
TESTS_64M := $(wildcard $(TEST_DIR)/rv64um-p-*.bin)
TESTS_64A := $(wildcard $(TEST_DIR)/rv64ua-p-*.bin)
TESTS_64F := $(wildcard $(TEST_DIR)/rv64uf-p-*.bin)
TESTS_64FX := $(filter-out $(TEST_DIR)/rv64uf-p-fcvt.bin $(TEST_DIR)/rv64uf-p-fcvt_w.bin, $(wildcard $(TEST_DIR)/rv64uf-p-*.bin))
TESTS_64D := $(wildcard $(TEST_DIR)/rv64ud-p-*.bin)

all:

run-simx-32imafd:
	$(foreach test, $(TESTS_32I) $(TESTS_32M) $(TESTS_32F) $(TESTS_32D) $(TESTS_32A), $(SIM_DIR)/simx/simx $(test) || exit;)

run-simx-64imafd:
	$(foreach test, $(TESTS_64I) $(TESTS_64M) $(TESTS_64F) $(TESTS_64D) $(TESTS_64A), $(SIM_DIR)/simx/simx $(test) || exit;)

run-simx-32i:
	$(foreach test, $(TESTS_32I), $(SIM_DIR)/simx/simx $(test) || exit;)

run-simx-32im:
	$(foreach test, $(TESTS_32I) $(TESTS_32M), $(SIM_DIR)/simx/simx $(test) || exit;)

run-simx-32f:
	$(foreach test, $(TESTS_32F), $(SIM_DIR)/simx/simx $(test) || exit;)

run-simx-32d:
	$(foreach test, $(TESTS_32D), $(SIM_DIR)/simx/simx $(test) || exit;)

run-simx-64im:
	$(foreach test, $(TESTS_64I) $(TESTS_64M), $(SIM_DIR)/simx/simx $(test) || exit;)

run-simx-64f:
	$(foreach test, $(TESTS_64F), $(SIM_DIR)/simx/simx $(test) || exit;)

run-simx-64fx:
	$(foreach test, $(TESTS_64FX), $(SIM_DIR)/simx/simx $(test) || exit;)

run-simx-64d:
	$(foreach test, $(TESTS_64D), $(SIM_DIR)/simx/simx $(test) || exit;)

run-simx-32: run-simx-32imafd

run-simx-64: run-simx-32imafd run-simx-64imafd

run-simx: run-simx-$(XLEN)

run-rtlsim-32imf:
	$(foreach test, $(TESTS_32I) $(TESTS_32M) $(TESTS_32F), $(SIM_DIR)/rtlsim/rtlsim $(test) || exit;)

run-rtlsim-64imf:
	$(foreach test, $(TESTS_64I) $(TESTS_64M) $(TESTS_64F), $(SIM_DIR)/rtlsim/rtlsim $(test) || exit;)

run-rtlsim-32i:
	$(foreach test, $(TESTS_32I), $(SIM_DIR)/rtlsim/rtlsim $(test) || exit;)

run-rtlsim-32im:
	$(foreach test, $(TESTS_32I) $(TESTS_32M), $(SIM_DIR)/rtlsim/rtlsim $(test) || exit;)

run-rtlsim-32f:
	$(foreach test, $(TESTS_32F), $(SIM_DIR)/rtlsim/rtlsim $(test) || exit;)

run-rtlsim-64im:
	$(foreach test, $(TESTS_64I) $(TESTS_64M), $(SIM_DIR)/rtlsim/rtlsim $(test) || exit;)

run-rtlsim-64f:
	$(foreach test, $(TESTS_64F), $(SIM_DIR)/rtlsim/rtlsim $(test) || exit;)

run-rtlsim-64fx:
	$(foreach test, $(TESTS_64FX), $(SIM_DIR)/rtlsim/rtlsim $(test) || exit;)

run-rtlsim-64d:
	$(foreach test, $(TESTS_64D), $(SIM_DIR)/rtlsim/rtlsim $(test) || exit;)

run-rtlsim-32: run-rtlsim-32imf

run-rtlsim-64: run-rtlsim-32imf run-rtlsim-64imf

run-rtlsim: run-rtlsim-$(XLEN)

clean: