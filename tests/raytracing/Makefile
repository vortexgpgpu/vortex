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
    rtu_smoke_bvh_instanced                                           \
    rtu_smoke_shadow rtu_smoke_cull_back                              \
    rtu_smoke_async_batch rtu_smoke_cull_mask                         \
    rtu_smoke_proc rtu_smoke_bvh6 rt_raycast

# --- common exclude list ---------------------------------------------
EXCLUDE :=

# --- per-backend exclude lists ---------------------------------------
# The RTL RTU runs the opaque trace path (flat WIDTH=0 + CW-BVH WIDTH=4/6),
# TLAS/instancing (flat-TLAS, CW-BVH LEAF_INST via the XformUnit), and the
# shader-callback / async-trap path (AHS/IS/CHS/MISS/SBT/proc) plus single-group
# reformation — all green on rtlsim.
# rtlsim-deferred = features/limits not yet in the RTL core:
#   - rtu_smoke_recursive  : in-trap nested-trace re-entrancy (RTU FSM)
#   - rtu_smoke_reform_mw  : multi-warp reformation
#   - rtu_smoke_reform_sbt : divergent-SBT multi-group reformation dispatch
#   - rtu_smoke_async_batch: §8.6 async ray pool (multiple traces in flight)
#   - rt_raycast           : RTU sustained multi-warp-load servicing deadlock
#                            (2 warps pass; the full framebuffer wedges the
#                            scoreboard on an unretired trace) — passes on simx.
# These remain simx-only until the next RTU RTL milestone closes them.
EXCLUDE_simx   :=
RTLSIM_DEFERRED := \
    rtu_smoke_recursive rtu_smoke_reform_mw rtu_smoke_reform_sbt \
    rtu_smoke_async_batch rt_raycast
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
