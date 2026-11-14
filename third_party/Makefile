all: cvfpu softfloat ramulator cocogfx

cvfpu:

SOFTFLOAT_DIR := softfloat/build/Linux-x86_64-GCC
SOFTFLOAT_OPTS := -fPIC -DSOFTFLOAT_ROUND_ODD -DINLINE_LEVEL=5 -DSOFTFLOAT_FAST_DIV32TO16 -DSOFTFLOAT_FAST_DIV64TO32 -DTHREAD_LOCAL=_Thread_local

# softfloat's own makefile does not track SOFTFLOAT_OPTS, so an archive built
# with older options (e.g. non-TLS softfloat_exceptionFlags) is otherwise kept.
softfloat:
	@echo '$(SOFTFLOAT_OPTS)' | cmp -s - softfloat.opts || $(MAKE) -C $(SOFTFLOAT_DIR) clean
	SPECIALIZE_TYPE=RISCV SOFTFLOAT_OPTS="$(SOFTFLOAT_OPTS)" $(MAKE) -C $(SOFTFLOAT_DIR)
	@echo '$(SOFTFLOAT_OPTS)' > softfloat.opts

ramulator/libramulator.so:
	cd ramulator && mkdir -p build && cd build && cmake .. && make -j4

# The GDDR6 device model Ramulator ships but does not build. It registers with
# Ramulator's factory when the library loads, built as Ramulator builds itself.
# Ramulator's cmake fetches spdlog and yaml-cpp into ramulator/ext, so the
# model compiles after it; otherwise the host's spdlog/fmt headers are used.
RAMULATOR_EXT_DIR := ramulator_ext
RAMULATOR_EXT_INCS := -isystem ramulator/src -isystem ramulator/ext/spdlog/include -isystem ramulator/ext/yaml-cpp/include
RAMULATOR_EXT_CXXFLAGS := -std=c++20 -O3 -DNDEBUG -fPIC $(RAMULATOR_EXT_INCS)

$(RAMULATOR_EXT_DIR)/gddr6.o: ramulator/src/dram/impl/GDDR6.cpp | ramulator/libramulator.so
	mkdir -p $(RAMULATOR_EXT_DIR)
	$(CXX) $(RAMULATOR_EXT_CXXFLAGS) -c $< -o $@

$(RAMULATOR_EXT_DIR)/libramulator-vortex.so: $(RAMULATOR_EXT_DIR)/gddr6.o ramulator/libramulator.so
	$(CXX) -shared $(filter %.o,$^) -Lramulator -lramulator -o $@

ramulator: ramulator/libramulator.so $(RAMULATOR_EXT_DIR)/libramulator-vortex.so

cocogfx/libcocogfx.a:
	$(MAKE) -C cocogfx
cocogfx: cocogfx/libcocogfx.a

clean:
	$(MAKE) -C $(SOFTFLOAT_DIR) clean
	rm -f softfloat.opts
	rm -rf ramulator/build ramulator/libramulator.so
	rm -f $(RAMULATOR_EXT_DIR)/*.o $(RAMULATOR_EXT_DIR)/*.so
	$(MAKE) -C cocogfx clean

.PHONY: all cvfpu softfloat ramulator cocogfx