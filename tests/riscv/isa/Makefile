include ../common.mk

# Upstream RISC-V ISA tests (riscv-tests). These are built on demand from
# the pinned upstream commit and run unmodified — Vortex's synchronous
# trap path (ECALL/EBREAK -> mtvec) lets the test's own trap_vector run,
# and the HTIF `tohost` watcher in the simulators harvests pass/fail.
#
# The upstream tree is cloned into the build directory (out-of-tree) and
# cached behind a stamp file: `make clean` preserves it, `make distclean`
# forces a re-fetch.

RISCV_TESTS_REPO   ?= https://github.com/riscv-software-src/riscv-tests.git
RISCV_TESTS_COMMIT ?= 1eb47d946c55f55cab8653c224c2993acc0276bd

UPSTREAM_DIR := $(ROOT_DIR)/tests/riscv/upstream
ISA_DIR      := $(UPSTREAM_DIR)/isa
STAMP        := $(UPSTREAM_DIR)/.installed

# Skip atomic ISA tests when the build is configured without EXT_A_ENABLE.
# Without atomics support, AMO instructions trap and the test fails. Resolve
# the post-preprocessor value of EXT_A_ENABLED rather than grepping (the
# header always lists both 0 and 1 branches inside an #ifdef block).
HAS_EXT_A := $(shell echo 'EXT_A_ENABLED' | gcc -E -P $(CONFIGS) -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -include VX_config.h - 2>/dev/null | tr -d ' \t\n')
HAS_EXT_C := $(shell echo 'EXT_C_ENABLED' | gcc -E -P $(CONFIGS) -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -include VX_config.h - 2>/dev/null | tr -d ' \t\n')

# Test-suite prefixes (extension-gated). A/C are only run when the build
# was configured with the corresponding extension.
P_32I := rv32ui-p
P_32M := rv32um-p
P_32F := rv32uf-p
P_32D := rv32ud-p
P_32A :=
P_32C :=
P_64I := rv64ui-p
P_64M := rv64um-p
P_64F := rv64uf-p
P_64D := rv64ud-p
P_64A :=
P_64C :=
ifneq ($(HAS_EXT_A),0)
P_32A := rv32ua-p
P_64A := rv64ua-p
endif
ifneq ($(HAS_EXT_C),0)
P_32C := rv32uc-p
P_64C := rv64uc-p
endif

# Run one or more test-suite prefixes against a simulator.
#   $(1) = simulator (simx | rtlsim)
#   $(2) = space-separated test prefixes (e.g. "rv32ui-p rv32um-p")
# Known-unsupported tests are skipped: fence_i (no instruction-cache
# coherence), ma_data (no misaligned-access trap), rv64uc-p-rvc
# (misaligned C.FLD the upstream test expects to trap on).
define run_isa
@set -e; for pfx in $(2); do for t in $(ISA_DIR)/$$pfx-*; do \
  case "$$t" in *.dump) continue;; esac; [ -f "$$t" ] || continue; \
  b=`basename $$t`; \
  case "$$b" in *-fence_i|*-ma_data|rv64uc-p-rvc) echo "[skip] $$b"; continue;; esac; \
  echo "[run]  $$b"; $(SIM_DIR)/$(1)/$(1) "$$t" || { echo "[FAIL] $$b"; exit 1; }; \
done; done; echo "[$(1)] passed: $(2)"
endef

all:

# Clone + build the upstream riscv-tests ISA suites for this build's XLEN.
$(STAMP):
	rm -rf $(UPSTREAM_DIR)
	git clone $(RISCV_TESTS_REPO) $(UPSTREAM_DIR)
	cd $(UPSTREAM_DIR) && git checkout --quiet $(RISCV_TESTS_COMMIT) && git submodule update --init --recursive
	PATH=$(RISCV_TOOLCHAIN_PATH)/bin:$$PATH $(MAKE) -C $(ISA_DIR) \
	  XLEN=$(XLEN) RISCV_PREFIX=$(RISCV_PREFIX)- \
	  rv$(XLEN)ui rv$(XLEN)um rv$(XLEN)uf rv$(XLEN)ud rv$(XLEN)ua rv$(XLEN)uc
	touch $@

install: $(STAMP)

run-simx-32imf: $(STAMP)
	$(call run_isa,simx,$(P_32I) $(P_32M) $(P_32F))

run-simx-64imf: $(STAMP)
	$(call run_isa,simx,$(P_64I) $(P_64M) $(P_64F))

run-simx-32imafd: $(STAMP)
	$(call run_isa,simx,$(P_32I) $(P_32M) $(P_32F) $(P_32D) $(P_32A) $(P_32C))

run-simx-64imafd: $(STAMP)
	$(call run_isa,simx,$(P_64I) $(P_64M) $(P_64F) $(P_64D) $(P_64A) $(P_64C))

run-simx-32i: $(STAMP)
	$(call run_isa,simx,$(P_32I))

run-simx-32im: $(STAMP)
	$(call run_isa,simx,$(P_32I) $(P_32M))

run-simx-32f: $(STAMP)
	$(call run_isa,simx,$(P_32F))

run-simx-32d: $(STAMP)
	$(call run_isa,simx,$(P_32D))

run-simx-32a: $(STAMP)
	$(call run_isa,simx,$(P_32A))

run-simx-32c: $(STAMP)
	$(call run_isa,simx,$(P_32C))

run-simx-64im: $(STAMP)
	$(call run_isa,simx,$(P_64I) $(P_64M))

run-simx-64f: $(STAMP)
	$(call run_isa,simx,$(P_64F))

run-simx-64d: $(STAMP)
	$(call run_isa,simx,$(P_64D))

run-simx-64a: $(STAMP)
	$(call run_isa,simx,$(P_64A))

run-simx-64c: $(STAMP)
	$(call run_isa,simx,$(P_64C))

run-simx-32: run-simx-32imafd

run-simx-64: run-simx-32imafd run-simx-64imafd

run-simx: run-simx-$(XLEN)

run-rtlsim-32imf: $(STAMP)
	$(call run_isa,rtlsim,$(P_32I) $(P_32M) $(P_32F))

run-rtlsim-64imf: $(STAMP)
	$(call run_isa,rtlsim,$(P_64I) $(P_64M) $(P_64F))

run-rtlsim-32i: $(STAMP)
	$(call run_isa,rtlsim,$(P_32I))

run-rtlsim-32im: $(STAMP)
	$(call run_isa,rtlsim,$(P_32I) $(P_32M))

run-rtlsim-32f: $(STAMP)
	$(call run_isa,rtlsim,$(P_32F))

run-rtlsim-32a: $(STAMP)
	$(call run_isa,rtlsim,$(P_32A))

run-rtlsim-32c: $(STAMP)
	$(call run_isa,rtlsim,$(P_32C))

run-rtlsim-64im: $(STAMP)
	$(call run_isa,rtlsim,$(P_64I) $(P_64M))

run-rtlsim-64f: $(STAMP)
	$(call run_isa,rtlsim,$(P_64F))

run-rtlsim-64d: $(STAMP)
	$(call run_isa,rtlsim,$(P_64D))

run-rtlsim-64a: $(STAMP)
	$(call run_isa,rtlsim,$(P_64A))

run-rtlsim-64c: $(STAMP)
	$(call run_isa,rtlsim,$(P_64C))

run-rtlsim-32: run-rtlsim-32imf

run-rtlsim-64: run-rtlsim-32imf run-rtlsim-64imf

run-rtlsim: run-rtlsim-$(XLEN)

# `clean` preserves the upstream cache; `distclean` forces a re-fetch.
clean:

distclean:
	rm -rf $(UPSTREAM_DIR)

.PHONY: all install clean distclean \
        run-simx run-simx-32 run-simx-64 run-rtlsim run-rtlsim-32 run-rtlsim-64
