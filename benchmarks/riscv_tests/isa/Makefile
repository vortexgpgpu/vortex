TESTS := $(wildcard *.hex)
#VTESTS := $(wildcard *-v-*.hex)
#TESTS := $(filter-out $(VTESTS) rv32ud-p-fclass.hex, $(TESTS))

run:
	cd ../../../hw/simulate/obj_dir && ./VVortex -f $(foreach test,$(TESTS),../../../benchmarks/riscv_tests/isa/$(test))
