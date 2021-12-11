ALL_TESTS := $(wildcard *.hex)

EXCLUDED_TESTS := rv64ud-p-move.hex

I_TESTS := $(wildcard *ui-p-*.hex)
M_TESTS := $(wildcard *um-p-*.hex)
F_TESTS := $(wildcard *uf-p-*.hex)
D_TESTS := $(filter-out $(EXCLUDED_TESTS), $(wildcard *ud-p-*.hex))

TESTS := $(I_TESTS) $(M_TESTS) $(F_TESTS) $(D_TESTS)

all:

run-simx-i:	
	$(foreach test, $(I_TESTS), ../../../sim/simx/simx -r -a rv64i -c 1 -i $(test) || exit;)

run-simx-m:	
	$(foreach test, $(M_TESTS), ../../../sim/simx/simx -r -a rv64im -c 1 -i $(test) || exit;)

run-simx-f:	
	$(foreach test, $(F_TESTS), ../../../sim/simx/simx -r -a rv64imf -c 1 -i $(test) || exit;)

run-simx-d:	
	$(foreach test, $(D_TESTS), ../../../sim/simx/simx -r -a rv64imfd -c 1 -i $(test) || exit;)

run-simx:	
	$(foreach test, $(TESTS), ../../../sim/simx/simx -r -a rv64i -c 1 -i $(test) || exit;)

run-rtlsim:
	$(foreach test, $(TESTS), ../../../sim/rtlsim/rtlsim -r $(test) || exit;)

clean: