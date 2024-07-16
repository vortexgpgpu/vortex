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
	@for test in  $(TESTS_32I) $(TESTS_32M) $(TESTS_32F) $(TESTS_32D) $(TESTS_32A); do $(SIM_DIR)/simx/simx $$test || exit 1; done

run-simx-64imafd:
	@for test in  $(TESTS_64I) $(TESTS_64M) $(TESTS_64F) $(TESTS_64D) $(TESTS_64A); do $(SIM_DIR)/simx/simx $$test || exit 1; done

run-simx-32i:
	@for test in  $(TESTS_32I); do $(SIM_DIR)/simx/simx $$test || exit 1; done

run-simx-32im:
	@for test in  $(TESTS_32I) $(TESTS_32M); do $(SIM_DIR)/simx/simx $$test || exit 1; done

run-simx-32f:
	@for test in  $(TESTS_32F); do $(SIM_DIR)/simx/simx $$test || exit 1; done

run-simx-32d:
	@for test in  $(TESTS_32D); do $(SIM_DIR)/simx/simx $$test || exit 1; done

run-simx-64im:
	@for test in  $(TESTS_64I) $(TESTS_64M); do $(SIM_DIR)/simx/simx $$test || exit 1; done

run-simx-64f:
	@for test in  $(TESTS_64F); do $(SIM_DIR)/simx/simx $$test || exit 1; done

run-simx-64fx:
	@for test in  $(TESTS_64FX); do $(SIM_DIR)/simx/simx $$test || exit 1; done

run-simx-64d:
	@for test in  $(TESTS_64D); do $(SIM_DIR)/simx/simx $$test || exit 1; done

run-simx-32: run-simx-32imafd

run-simx-64: run-simx-32imafd run-simx-64imafd

run-simx: run-simx-$(XLEN)

run-rtlsim-32imf:
	@for test in  $(TESTS_32I) $(TESTS_32M) $(TESTS_32F); do $(SIM_DIR)/rtlsim/rtlsim $$test || exit 1; done

run-rtlsim-64imf:
	@for test in  $(TESTS_64I) $(TESTS_64M) $(TESTS_64F); do $(SIM_DIR)/rtlsim/rtlsim $$test || exit 1; done

run-rtlsim-32i:
	@for test in  $(TESTS_32I); do $(SIM_DIR)/rtlsim/rtlsim $$test || exit 1; done

run-rtlsim-32im:
	@for test in  $(TESTS_32I) $(TESTS_32M); do $(SIM_DIR)/rtlsim/rtlsim $$test || exit 1; done

run-rtlsim-32f:
	@for test in  $(TESTS_32F); do $(SIM_DIR)/rtlsim/rtlsim $$test || exit 1; done

run-rtlsim-64im:
	@for test in  $(TESTS_64I) $(TESTS_64M); do $(SIM_DIR)/rtlsim/rtlsim $$test || exit 1; done

run-rtlsim-64f:
	@for test in  $(TESTS_64F); do $(SIM_DIR)/rtlsim/rtlsim $$test || exit 1; done

run-rtlsim-64fx:
	@for test in  $(TESTS_64FX); do $(SIM_DIR)/rtlsim/rtlsim $$test || exit 1; done

run-rtlsim-64d:
	@for test in  $(TESTS_64D); do $(SIM_DIR)/rtlsim/rtlsim $$test || exit 1; done

run-rtlsim-32: run-rtlsim-32imf

run-rtlsim-64: run-rtlsim-32imf run-rtlsim-64imf

run-rtlsim: run-rtlsim-$(XLEN)

clean: