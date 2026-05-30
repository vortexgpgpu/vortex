ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# --- master list ------------------------------------------------------
TESTS := \
    basic demo dogfood dropout dotproduct dotproduct2 mstress io_addr \
    arith printf diverge sort fence                                   \
    vecadd vecadd_v1 sgemm sgemmx sgemm_v1 conv3 relu sgemv           \
    sgemm2 sgemm2_v1 madmax stencil3d raycast bfs jacobi pathfinder   \
    softmax                                                           \
    sgemm_tcu sgemm_tcu_wg sgemm_tcu_sp sgemm_tcu_wg_sp               \
    sgemm2_dxa sgemm2_tcu sgemm_tcu_wg_dxa sgemm_tcu_wg_sp_dxa        \
    sgemm2_dxa_mcast sgemm_tcu_wg_dxa_mcast                           \
    dxa_copy dxa_copy_mcast                                           \
    rtu_smoke rtu_smoke_ahs rtu_smoke_reform rtu_smoke_reform_mw      \
    rtu_smoke_reform_sbt rtu_smoke_multitri rtu_smoke_chs             \
    rtu_smoke_miss rtu_smoke_is                                       \
    mpi_diverge mpi_neighbor_a2a_conv3 mpi_conv3                      \
    async_barrier async_gbarrier packld wgather wsync                 \
    occupancy multikernel

# --- common exclude list ---------------------------------------------
EXCLUDE := \
    mpi_diverge \
    mpi_neighbor_a2a_conv3 \
    mpi_conv3

# --- per-backend exclude lists ---------------------------------------
EXCLUDE_simx   :=
EXCLUDE_rtlsim :=
EXCLUDE_opae   :=
EXCLUDE_xrt    :=

BACKENDS := simx rtlsim opae xrt

# --- aggregate targets ------------------------------------------------
.PHONY: all clean $(addprefix run-,$(BACKENDS))

# `all` and every `run-<backend>` honor BOTH the common EXCLUDE and the
# backend-specific EXCLUDE_<backend>. `clean` walks the FULL list so a
# disabled tree still gets scrubbed.
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

# run-<backend>-<test>: split "<backend>-<test>" on the first dash.
# Subdir names must not contain '-' (none currently do).
run-%:
	$(MAKE) -C $(word 2,$(subst -, ,$*)) run-$(word 1,$(subst -, ,$*))
