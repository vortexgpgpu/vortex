ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

INC_DIR := $(VORTEX_HOME)/sw/runtime/include
# libvortex.so + libvortex-<driver>.so live in the build tree's sw/runtime;
# ROOT_DIR is the build directory (realpath ../..) so this works for any
# build dir name (build, build32, build64).
RT_DIR  := $(ROOT_DIR)/sw/runtime

# test_module_kernel exercises a real kernel launch via vx_kernel_h; point
# it at the `basic` regression kernel image. The runtime() target in
# ci/regression.sh builds this before invoking run-simx.
VXBIN := $(ROOT_DIR)/tests/regression/basic/kernel.vxbin

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors -Werror
CXXFLAGS += -O2 -DNDEBUG
CXXFLAGS += -I$(INC_DIR) -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw
# Resolve the toml + CONFIGS overrides into the canonical -D... list, the
# same way sim/simx/Makefile does. Project as -DVX_CFG_* flags.
XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py --config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS) -DVX_CFG_XLEN=$(XLEN)')
CXXFLAGS += $(XCONFIGS)

LDFLAGS += -Wl,-rpath,$(RT_DIR) -L$(RT_DIR) -lvortex -pthread

TESTS := test_basic test_async test_timeline_events test_module_kernel test_rect_fill

.PHONY: all run run-simx run-xrt clean

all: $(TESTS)

test_basic: $(VORTEX_HOME)/tests/runtime/test_basic.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

test_async: $(VORTEX_HOME)/tests/runtime/test_async.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

test_timeline_events: $(VORTEX_HOME)/tests/runtime/test_timeline_events.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

test_module_kernel: $(VORTEX_HOME)/tests/runtime/test_module_kernel.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

test_rect_fill: $(VORTEX_HOME)/tests/runtime/test_rect_fill.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

# Bare run — honours whatever VORTEX_DRIVER is in the environment.
run: $(TESTS)
	@for t in $(TESTS); do \
	  echo "[RUN] $$t"; \
	  ./$$t || exit 1; \
	done

# CI entry points — pin the backend driver + locate the runtime libs and
# the test kernel image. VX_TEST_VXBIN is consumed by test_module_kernel.
run-simx: $(TESTS)
	@for t in $(TESTS); do \
	  echo "[RUN] $$t (simx)"; \
	  VORTEX_DRIVER=simx LD_LIBRARY_PATH=$(RT_DIR) \
	  VX_TEST_VXBIN=$(VXBIN) ./$$t || exit 1; \
	done

run-xrt: $(TESTS)
	@for t in $(TESTS); do \
	  echo "[RUN] $$t (xrt)"; \
	  VORTEX_DRIVER=xrt LD_LIBRARY_PATH=$(RT_DIR) \
	  VX_TEST_VXBIN=$(VXBIN) ./$$t || exit 1; \
	done

clean:
	rm -f $(TESTS)
