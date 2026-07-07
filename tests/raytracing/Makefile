ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# --- master list ------------------------------------------------------
# All RTU smoke tests. Same shape as tests/regression/Makefile; only
# the test list differs. See ../regression/common.mk for the
# per-test build/run rules (shared via include below).
TESTS := \
    rtu_smoke rtu_smoke_ahs rtu_smoke_reform rtu_smoke_reform_mw      \
    rtu_smoke_reform_sbt rtu_smoke_multitri rtu_smoke_chs             \
    rtu_smoke_miss rtu_smoke_is rtu_smoke_sbt rtu_smoke_tlas          \
    rtu_smoke_ahs_mixed rtu_smoke_recursive                           \
    rtu_smoke_bvh_basic rtu_smoke_bvh_multilevel                      \
    rtu_smoke_bvh_instanced rtu_smoke_ahs_bvh                         \
    rtu_smoke_shadow rtu_smoke_cull_back                              \
    rtu_smoke_async_batch rtu_smoke_cull_mask                         \
    rtu_smoke_proc rtu_smoke_bvh6 rtu_bvh_multinode rtu_smoke_numctx   \
    rtu_smoke_inst_flags rtu_smoke_tlas_builder rtu_smoke_deep_stack       \
    rtu_smoke_ahs_custom rtu_smoke_fat_leaf rtu_smoke_ahs_geom             \
    rtu_smoke_deep_tlas rt_raycast

# --- common exclude list ---------------------------------------------
EXCLUDE :=

# --- per-backend exclude lists ---------------------------------------
# The RTL RTU runs the opaque trace path (flat WIDTH=0 + CW-BVH WIDTH=4/6),
# TLAS/instancing (flat-TLAS, CW-BVH LEAF_INST via the XformUnit), and the
# shader-callback / async-trap path (AHS/IS/CHS/MISS/SBT/proc) plus single-group
# reformation — all green on rtlsim.
# rt_raycast and rtu_bvh_multinode issue one trace per thread and immediately
# wait on it (a single trace in flight per warp): they serialise through the
# single-trace RTU core but each completes, so they are green on rtlsim (full
# framebuffer + 36-node BVH both validated) and run there now.
# rtlsim-deferred = features/limits not yet in the RTL core. These all require
# MULTIPLE traces in flight concurrently — the §8.6 async ray pool the RTL core
# does not yet implement (it services one trace-warp at a time):
#   - rtu_smoke_recursive  : in-trap nested-trace re-entrancy (RTU FSM)
#   - rtu_smoke_reform_mw  : multi-warp reformation (concurrent traces)
#   - rtu_smoke_reform_sbt : divergent-SBT multi-group reformation dispatch
#   - rtu_smoke_async_batch: §8.6 async ray pool — 4 vx_rt_wtrace issued
#                            back-to-back before any wait (4 traces in flight)
# These remain simx-only until the RTU core grows the bounded N-slot in-flight
# trace pool (SimX §8.6 semantics).
EXCLUDE_simx   :=
RTLSIM_DEFERRED := \
    rtu_smoke_recursive rtu_smoke_reform_mw rtu_smoke_reform_sbt \
    rtu_smoke_async_batch
EXCLUDE_rtlsim := $(RTLSIM_DEFERRED)
EXCLUDE_opae   := $(TESTS)
EXCLUDE_xrt    := $(TESTS)

BACKENDS := simx rtlsim opae xrt

# --- aggregate targets ------------------------------------------------
.PHONY: all clean $(addprefix run-,$(BACKENDS))

ACTIVE_TESTS = $(filter-out $(EXCLUDE),$(TESTS))
backend_tests = $(filter-out $(EXCLUDE) $(EXCLUDE_$(1)),$(TESTS))

all:   $(addprefix build-,$(ACTIVE_TESTS))
clean: $(addprefix clean-,$(TESTS))

define backend_rule
.PHONY: run-$(1)
run-$(1): $$(addprefix run-$(1)-,$$(call backend_tests,$(1)))
endef
$(foreach b,$(BACKENDS),$(eval $(call backend_rule,$(b))))

# --- pattern recipes --------------------------------------------------
.PHONY: build-% clean-%

build-%:
	$(MAKE) -C $* all

clean-%:
	$(MAKE) -C $* clean

run-%:
	$(MAKE) -C $(word 2,$(subst -, ,$*)) run-$(word 1,$(subst -, ,$*))
