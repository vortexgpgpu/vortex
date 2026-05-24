# Shared configuration for the riscv-tests suites (isa + benchmarks).
#
# Resolve the repo/build root from this file's own location so the same
# common.mk works whether it is included from tests/riscv/ or from
# tests/riscv/{isa,benchmarks}/.
COMMON_MK_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
ROOT_DIR := $(realpath $(COMMON_MK_DIR)/../..)
include $(ROOT_DIR)/config.mk

SIM_DIR := $(ROOT_DIR)/sim

# Upstream riscv-tests. The ISA suites and benchmarks are cloned and built
# on demand from a pinned commit, out-of-tree under the build directory,
# and cached behind a stamp file. `make clean` keeps the cache; the build
# is forced fresh by removing $(RISCV_TESTS_DIR).
RISCV_TESTS_REPO   ?= https://github.com/riscv-software-src/riscv-tests.git
RISCV_TESTS_COMMIT ?= 1eb47d946c55f55cab8653c224c2993acc0276bd

RISCV_TESTS_DIR   := $(ROOT_DIR)/tests/riscv/upstream
ISA_DIR           := $(RISCV_TESTS_DIR)/isa
BENCHMARKS_DIR    := $(RISCV_TESTS_DIR)/benchmarks
RISCV_TESTS_STAMP := $(RISCV_TESTS_DIR)/.installed

# The upstream benchmarks default to a hard-double ABI (ilp32d/lp64d);
# the Vortex riscv toolchain ships a single non-multilib runtime, so the
# benchmarks must be built for its ABI. Restrict to the scalar suite
# (the mt-*/pmp/vec-* benchmarks need features Vortex lacks).
ifeq ($(XLEN),64)
  BENCH_ABI   ?= lp64d
  BENCH_MARCH ?= rv64imafd
else
  BENCH_ABI   ?= ilp32f
  BENCH_MARCH ?= rv32imf_zicsr
endif
BENCH_LIST := median qsort rsort towers memcpy multiply dhrystone spmv

# Clone + build the upstream riscv-tests ISA suites and benchmarks for
# this build's XLEN. Uses the riscv toolchain installed under $(TOOLDIR).
$(RISCV_TESTS_STAMP):
	rm -rf $(RISCV_TESTS_DIR)
	git clone $(RISCV_TESTS_REPO) $(RISCV_TESTS_DIR)
	cd $(RISCV_TESTS_DIR) && git checkout --quiet $(RISCV_TESTS_COMMIT) && git submodule update --init --recursive
	# Benchmark-only patch: route console output / exit through Vortex
	# MMIO instead of HTIF (the ISA tests are built unmodified).
	cd $(RISCV_TESTS_DIR) && git apply $(VORTEX_HOME)/miscs/patches/riscv-benchmarks.patch
	PATH=$(RISCV_TOOLCHAIN_PATH)/bin:$$PATH $(MAKE) -C $(ISA_DIR) \
	  XLEN=$(XLEN) RISCV_PREFIX=$(RISCV_PREFIX)- \
	  rv$(XLEN)ui rv$(XLEN)um rv$(XLEN)uf rv$(XLEN)ud rv$(XLEN)ua rv$(XLEN)uc
	# The patched crt.S/syscalls.c reference VX_MEM_IO_* symbols from
	# the generated sw/VX_types.h; inject that include path into the
	# upstream Makefile via RISCV_GCC (the only compile entry point).
	PATH=$(RISCV_TOOLCHAIN_PATH)/bin:$$PATH $(MAKE) -C $(BENCHMARKS_DIR) \
	  XLEN=$(XLEN) RISCV_PREFIX=$(RISCV_PREFIX)- src_dir=$(BENCHMARKS_DIR) \
	  ABI=$(BENCH_ABI) RISCV_MARCH=$(BENCH_MARCH) \
	  RISCV_GCC="$(RISCV_PREFIX)-gcc -I$(ROOT_DIR)/sw" \
	  $(addsuffix .riscv,$(BENCH_LIST))
	touch $@

install: $(RISCV_TESTS_STAMP)

distclean-upstream:
	rm -rf $(RISCV_TESTS_DIR)

.PHONY: install distclean-upstream
