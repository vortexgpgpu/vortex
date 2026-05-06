all: cvfpu softfloat ramulator cocogfx

cvfpu:

softfloat:
	SPECIALIZE_TYPE=RISCV SOFTFLOAT_OPTS="-fPIC -DSOFTFLOAT_ROUND_ODD -DINLINE_LEVEL=5 -DSOFTFLOAT_FAST_DIV32TO16 -DSOFTFLOAT_FAST_DIV64TO32" $(MAKE) -C softfloat/build/Linux-x86_64-GCC

ramulator/libramulator.so:
	cd ramulator && mkdir -p build && cd build && cmake .. && make -j4
ramulator: ramulator/libramulator.so

cocogfx/libcocogfx.a:
	$(MAKE) -C cocogfx
cocogfx: cocogfx/libcocogfx.a

clean:
	$(MAKE) -C softfloat/build/Linux-x86_64-GCC clean
	rm -rf ramulator/build ramulator/libramulator.so
	$(MAKE) -C cocogfx clean

.PHONY: all cvfpu softfloat ramulator cocogfx