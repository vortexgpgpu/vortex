ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# --- master list ------------------------------------------------------
# All RTU smoke tests. Same shape as tests/regression/Makefile; only
# the test list differs. See ../regression/common.mk for the
# per-test build/run rules (shared via include below).
TESTS := \
    rt_smoke rt_smoke_ahs rt_smoke_reform rt_smoke_reform_mw      \
    rt_smoke_reform_sbt rt_smoke_multitri rt_smoke_chs             \
    rt_smoke_miss rt_smoke_is rt_smoke_sbt rt_smoke_tlas          \
    rt_smoke_ahs_mixed rt_smoke_recursive                           \
    rt_smoke_bvh_basic rt_smoke_bvh_multilevel                      \
    rt_smoke_bvh_instanced rt_smoke_ahs_bvh                         \
    rt_smoke_shadow rt_smoke_cull_back                              \
    rt_smoke_async_batch rt_smoke_cull_mask                         \
    rt_smoke_proc rt_smoke_bvh6 rt_bvh_multinode rt_smoke_numctx   \
    rt_smoke_inst_flags rt_smoke_tlas_builder rt_smoke_deep_stack       \
    rt_smoke_ahs_custom rt_smoke_fat_leaf rt_smoke_ahs_geom             \
    rt_smoke_host_cfg rt_smoke_deep_tlas rt_raycast

# --- common exclude list ---------------------------------------------
EXCLUDE :=

# --- per-backend exclude lists ---------------------------------------
# The RTL RTU runs the opaque trace path (flat WIDTH=0 + CW-BVH WIDTH=4/6),
# TLAS/instancing (flat-TLAS, CW-BVH LEAF_INST via the XformUnit), and the
# candidate-return shader path (AHS/IS/CHS/MISS/SBT/proc — a returned candidate
# is serviced inline by the warp's CONTINUE loop) plus single-group reformation
# — all green on rtlsim.
# rt_raycast and rt_bvh_multinode issue one trace per thread and immediately
# wait on it (a single trace in flight per warp): they serialise through the
# single-trace RTU core but each completes, so they are green on rtlsim (full
# framebuffer + 36-node BVH both validated) and run there now.
# rtlsim-deferred = features/limits not yet in the RTL core. These all require
# MULTIPLE traces in flight concurrently — the §8.6 async ray pool the RTL core
# does not yet implement (it services one trace-warp at a time):
#   - rt_smoke_recursive  : nested-trace re-entrancy needs a per-warp window
#                            context stack (single-context window on RTL)
#   - rt_smoke_reform_mw  : multi-warp reformation (concurrent traces)
#   - rt_smoke_reform_sbt : divergent-SBT multi-group reformation dispatch
#   - rt_smoke_async_batch: §8.6 async ray pool — 4 vx_rt_wtrace issued
#                            back-to-back before any wait (4 traces in flight)
# These remain simx-only until the RTU core grows the bounded N-slot in-flight
# trace pool (SimX §8.6 semantics).
EXCLUDE_simx   :=
RTLSIM_DEFERRED := \
    rt_smoke_recursive rt_smoke_reform_mw rt_smoke_reform_sbt \
    rt_smoke_async_batch
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
