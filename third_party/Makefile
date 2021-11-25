all: fpnew cocogfx softfloat

fpnew:

cocogfx:
	$(MAKE) -C cocogfx

softfloat:
	SPECIALIZE_TYPE=RISCV SOFTFLOAT_OPTS="-fPIC -DSOFTFLOAT_ROUND_ODD -DINLINE_LEVEL=5 -DSOFTFLOAT_FAST_DIV32TO16 -DSOFTFLOAT_FAST_DIV64TO32" $(MAKE) -C softfloat/build/Linux-x86_64-GCC

clean:
	$(MAKE) clean -C cocogfx
	$(MAKE) -C softfloat/build/Linux-x86_64-GCC clean

.PHONY: all fpnew cocogfx softfloat