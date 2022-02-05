XLEN ?= 32

SIM_DIR=../../../sim

TESTS_32I := $(filter-out rv32ui-p-fence_i.hex, $(wildcard rv32ui-p-*.hex))
TESTS_32M := $(wildcard rv32um-p-*.hex)
TESTS_32F := $(wildcard rv32uf-p-*.hex)
TESTS_32D := $(wildcard rv32ud-p-*.hex)

TESTS_64I := $(wildcard rv64ui-p-*.hex)
TESTS_64M := $(wildcard rv64um-p-*.hex)
TESTS_64F := $(wildcard rv64uf-p-*.hex)
TESTS_64D := $(wildcard rv64ud-p-*.hex)

all:

run-simx-32i:
	$(foreach test, $(TESTS_32I), $(SIM_DIR)/simx/simx -r -c 1 -i $(test) || exit;)

run-simx-32im:
	$(foreach test,  $(TESTS_32I) $(TESTS_32M), $(SIM_DIR)/simx/simx -r -c 1 -i $(test) || exit;)

run-simx-32imf:
	$(foreach test, $(TESTS_32I) $(TESTS_32M) $(TESTS_32F), $(SIM_DIR)/simx/simx -r -c 1 -i $(test) || exit;)

run-simx-32imfd:
	$(foreach test, $(TESTS_32I) $(TESTS_32M) $(TESTS_32F) $(TESTS_32D), $(SIM_DIR)/simx/simx -r -c 1 -i $(test) || exit;)

run-simx-64i:
	$(foreach test, $(TESTS_64I), $(SIM_DIR)/simx/simx -r -c 1 -i $(test) || exit;)

run-simx-64im:
	$(foreach test, $(TESTS_64I) $(TESTS_64M), $(SIM_DIR)/simx/simx -r -c 1 -i $(test) || exit;)

run-simx-64imf:
	$(foreach test, $(TESTS_64I) $(TESTS_64M) $(TESTS_64F), $(SIM_DIR)/simx/simx -r -c 1 -i $(test) || exit;)

run-simx-64imfd:
	$(foreach test, $(TESTS_64I) $(TESTS_64M) $(TESTS_64F) $(TESTS_64D), $(SIM_DIR)/simx/simx -r -c 1 -i $(test) || exit;)

run-simx-32: run-simx-32imfd

run-simx-64: run-simx-32imfd run-simx-64imfd

run-simx: run-simx-$(XLEN)

run-rtlsim:
	$(foreach test, $(TESTS_32I) $(TESTS_32M) $(TESTS_32F), $(SIM_DIR)/rtlsim/rtlsim -r $(test) || exit;)

clean: