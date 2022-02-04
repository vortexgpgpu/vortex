ALL_TESTS     := $(wildcard *.hex)
ALL_TESTS_32  := $(wildcard rv32*.hex)
ALL_TESTS_64 := $(wildcard rv64*.hex)


D_TESTS_32 := $(wildcard rv32ud-p-*.hex)

EXCLUDED_TESTS_32 := $(V_TESTS) $(D_TESTS_32)
EXCLUDED_TESTS_64 := rv64ud-p-recoding.hex

TESTS_32 := $(filter-out $(EXCLUDED_TESTS_32), $(ALL_TESTS_32))
TESTS_64 := $(filter-out $(EXCLUDED_TESTS_64), $(ALL_TESTS_64))

### To be deleted later
32I := $(wildcard rv32ui-p-*.hex)
32M := $(wildcard rv32um-p-*.hex)
32F := $(wildcard rv32uf-p-*.hex)
32D := $(wildcard rv32ud-p-*.hex)

64I := $(wildcard rv64ui-p-*.hex)
64M := $(wildcard rv64um-p-*.hex)
64F := $(wildcard rv64uf-p-*.hex)
64D := $(filter-out rv64ud-p-recoding.hex rv64ud-p-ldst.hex, $(wildcard rv64ud-p-*.hex))
###

all:

### To be deleted later
32i:
	$(foreach test, $(32I), ../../../sim/simx/simx -r -a rv32i -c 1 -i $(test) || exit;)

32m:
	$(foreach test, $(32M), ../../../sim/simx/simx -r -a rv32im -c 1 -i $(test) || exit;)

32f:
	$(foreach test, $(32F), ../../../sim/simx/simx -r -a rv32imf -c 1 -i $(test) || exit;)

32d:
	$(foreach test, $(32D), ../../../sim/simx/simx -r -a rv32imfd -c 1 -i $(test) || exit;)

64i:
	$(foreach test, $(64I), ../../../sim/simx/simx -r -a rv64i -c 1 -i $(test) || exit;)

64m:
	$(foreach test, $(64M), ../../../sim/simx/simx -r -a rv64im -c 1 -i $(test) || exit;)

64f:
	$(foreach test, $(64F), ../../../sim/simx/simx -r -a rv64imf -c 1 -i $(test) || exit;)

64d:
	$(foreach test, $(64D), ../../../sim/simx/simx -r -a rv64imfd -c 1 -i $(test) || exit;)
###

run-simx: run-simx-32imf

run-simx-32imf:	
	$(foreach test, $(TESTS_32), ../../../sim/simx/simx -r -a rv32imf -c 1 -i $(test) || exit;)

run-simx-32imfd:	
	$(foreach test, $(TESTS_32) $(D_TESTS_32), ../../../sim/simx/simx -r -a rv32imfd -c 1 -i $(test) || exit;)

run-simx-64imfd:
	$(foreach test, $(TESTS_64), ../../../sim/simx/simx -r -a rv64imfd -c 1 -i $(test) || exit;)

run-rtlsim:
	$(foreach test, $(TESTS_32), ../../../sim/rtlsim/rtlsim -r $(test) || exit;)

clean: