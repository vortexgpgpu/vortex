ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# --- master list ------------------------------------------------------
TESTS := \
    vecadd sgemm conv3 psort saxpy sfilter sgemm2 sgemm3 psum oclprintf \
    dotproduct transpose spmv stencil lbm nearn guassian kmeans         \
    blackscholes bfs copybuf

# --- common exclude list ---------------------------------------------
EXCLUDE := \
    transpose \
    kmeans    \
    bfs       \
    lbm       \
    copybuf

# --- per-backend exclude lists ---------------------------------------
EXCLUDE_simx   :=
EXCLUDE_rtlsim :=
EXCLUDE_opae   :=
EXCLUDE_xrt    :=

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
