include ../common.mk

# Upstream riscv-tests benchmarks (dhrystone, qsort, towers, ...). The
# upstream tree is cloned and built by the $(RISCV_TESTS_STAMP) rule in
# ../common.mk; the run-* targets below depend on it.
#
# The benchmarks use HTIF syscalls (console output) and the HTIF `tohost`
# exit word; the simulators' HostMonitor services both.

define run_bench
@set -e; for t in $(BENCHMARKS_DIR)/*.riscv; do \
  case "$$t" in *.dump) continue;; esac; [ -f "$$t" ] || continue; \
  b=`basename $$t`; \
  echo "[run]  $$b"; $(SIM_DIR)/$(1)/$(1) "$$t" || { echo "[FAIL] $$b"; exit 1; }; \
done; echo "[$(1)] benchmarks passed"
endef

all:

run-simx: $(RISCV_TESTS_STAMP)
	$(call run_bench,simx)

run-rtlsim: $(RISCV_TESTS_STAMP)
	$(call run_bench,rtlsim)

clean:

.PHONY: all run-simx run-rtlsim clean
