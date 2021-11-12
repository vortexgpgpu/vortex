ALL_TESTS := $(wildcard *.hex)

D_TESTS := $(wildcard *ud-p-*.hex)
V_TESTS := $(wildcard *-v-*.hex)

EXCLUDED_TESTS := $(V_TESTS) $(D_TESTS) rv32si-p-scall.hex rv32si-p-sbreak.hex rv32mi-p-breakpoint.hex rv32mi-p-ma_addr.hex rv32mi-p-mcsr.hex rv32mi-p-ma_fetch.hex rv32mi-p-csr.hex rv32si-p-dirty.hex rv32ui-p-fence_i.hex rv32si-p-csr.hex rv32mi-p-shamt.hex rv32si-p-wfi.hex rv32si-p-ma_fetch.hex rv32mi-p-illegal.hex rv32uc-p-rvc.hex rv32mi-p-sbreak.hex

TESTS := $(filter-out $(EXCLUDED_TESTS), $(ALL_TESTS))

all:

run-simx:	
	$(foreach test, $(TESTS), ../../../sim/simX/simX -r -a rv32i -c 1 -i $(test) || exit;)

run-rtlsim:
	$(foreach test, $(TESTS), ../../../sim/rtlsim/rtlsim -r $(test) || exit;)

clean:
