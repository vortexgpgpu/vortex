ALL_TESTS := $(wildcard *.hex)

D_TESTS := $(wildcard *ud-p-*.hex)
V_TESTS := $(wildcard *-v-*.hex)

EXCLUDED_TESTS := $(V_TESTS) $(D_TESTS) rv32si-p-scall.hex rv32si-p-sbreak.hex rv32mi-p-breakpoint.hex rv32ua-p-amomax_w.hex rv32ua-p-amoxor_w.hex rv32ua-p-amoor_w.hex rv32mi-p-ma_addr.hex rv32mi-p-mcsr.hex rv32ua-p-amoswap_w.hex rv32mi-p-ma_fetch.hex rv32mi-p-csr.hex rv32ua-p-amoadd_w.hex rv32si-p-dirty.hex rv32ui-p-fence_i.hex rv32si-p-csr.hex rv32mi-p-shamt.hex rv32ua-p-amomin_w.hex rv32ua-p-lrsc.hex rv32si-p-wfi.hex rv32ua-p-amomaxu_w.hex rv32si-p-ma_fetch.hex rv32mi-p-illegal.hex rv32uc-p-rvc.hex rv32mi-p-sbreak.hex rv32ua-p-amominu_w.hex rv32ua-p-amoand_w.hex

TESTS := $(filter-out $(EXCLUDED_TESTS), $(ALL_TESTS))

all:

run-simx:	
	$(foreach test, $(TESTS), ../../../sim/simX/simX -r -a rv32i -c 1 -i $(test) || exit;)

run-rtlsim:
	$(foreach test, $(TESTS), ../../../sim/rtlsim/rtlsim -r $(test) || exit;)

clean: