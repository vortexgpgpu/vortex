all: fpnew softfloat ramulator

fpnew:

softfloat:
	SPECIALIZE_TYPE=RISCV SOFTFLOAT_OPTS="-fPIC -DSOFTFLOAT_ROUND_ODD -DINLINE_LEVEL=5 -DSOFTFLOAT_FAST_DIV32TO16 -DSOFTFLOAT_FAST_DIV64TO32" $(MAKE) -C softfloat/build/Linux-x86_64-GCC

ramulator:
	cd ramulator && git apply ../../miscs/patch/ramulator.patch 2> /dev/null; true
	$(MAKE) -C ramulator libramulator.a

clean:
	$(MAKE) -C softfloat/build/Linux-x86_64-GCC clean
	$(MAKE) -C ramulator clean

.PHONY: all fpnew softfloat ramulator