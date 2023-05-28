# TODO: Missing features
# - unaligned LD/ST (RTL)
# - 64-bit float extension (RTL)
# - 64-bit double extension (RTL)
# - compressed extension
# - fence extension
# - atomics extension
# - vector extension

XLEN ?= 32

SIM_DIR=../../../sim

TESTS_32I := $(filter-out rv32ui-p-ma_data.hex rv32ui-p-fence_i.hex, $(wildcard rv32ui-p-*.hex))
TESTS_32M := $(wildcard rv32um-p-*.hex)
TESTS_32A := $(wildcard rv32ua-p-*.hex)
TESTS_32F := $(wildcard rv32uf-p-*.hex)
TESTS_32D := $(wildcard rv32ud-p-*.hex)

TESTS_64I := $(filter-out rv64ui-p-ma_data.hex rv64ui-p-fence_i.hex, $(wildcard rv64ui-p-*.hex))
TESTS_64M := $(wildcard rv64um-p-*.hex)
TESTS_64A := $(wildcard rv64ua-p-*.hex)
TESTS_64F := $(wildcard rv64uf-p-*.hex)
TESTS_64D := $(wildcard rv64ud-p-*.hex)

all:

run-simx-32imfd:
	$(foreach test, $(TESTS_32I) $(TESTS_32M) $(TESTS_32F) $(TESTS_32D) $(TESTS_32A), $(SIM_DIR)/simx/simx -r -c 1 -i $(test) || exit;)

run-simx-64imfd:
	$(foreach test, $(TESTS_64I) $(TESTS_64M) $(TESTS_64F) $(TESTS_64D) $(TESTS_64A), $(SIM_DIR)/simx/simx -r -c 1 -i $(test) || exit;)

run-simx-32: run-simx-32imfd

run-simx-64: run-simx-32imfd run-simx-64imfd

run-simx: run-simx-$(XLEN)

run-rtlsim-32imf:
	$(foreach test, $(TESTS_32I) $(TESTS_32M) $(TESTS_32F), $(SIM_DIR)/rtlsim/rtlsim -r $(test) || exit;)

run-rtlsim-64im:
	$(foreach test, $(TESTS_64I) $(TESTS_64M), $(SIM_DIR)/rtlsim/rtlsim -r $(test) || exit;)

run-rtlsim-32: run-rtlsim-32imf

run-rtlsim-64: run-rtlsim-32imf run-rtlsim-64im

run-rtlsim: run-rtlsim-$(XLEN)

clean: