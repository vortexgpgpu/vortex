ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# --- master list ------------------------------------------------------
TESTS := conform hello fibonacci vecadd

# --- common exclude list ---------------------------------------------
EXCLUDE :=

# --- per-backend exclude lists ---------------------------------------
EXCLUDE_simx   :=
EXCLUDE_rtlsim :=

BACKENDS := simx rtlsim

# --- aggregate targets ------------------------------------------------
.PHONY: all clean $(addprefix run-,$(BACKENDS))

ACTIVE_TESTS = $(filter-out $(EXCLUDE),$(TESTS))
backend_tests = $(filter-out $(EXCLUDE) $(EXCLUDE_$(1)),$(TESTS))

all:   $(addprefix build-,$(ACTIVE_TESTS))
clean: $(addprefix clean-,$(TESTS))

# The per-test runs build a missing simulator on demand; sim-<backend> builds
# it once first, so parallel runs don't each start the same build.
define backend_rule
.PHONY: run-$(1) sim-$(1)
run-$(1): $$(addprefix run-$(1)-,$$(call backend_tests,$(1)))
sim-$(1):
	@test -e $$(ROOT_DIR)/sim/$(1)/$(1) || $$(MAKE) -C $$(ROOT_DIR)/sim/$(1)
run-$(1)-%: kernel-lib sim-$(1)
	$$(MAKE) -C $$* run-$(1)
endef
$(foreach b,$(BACKENDS),$(eval $(call backend_rule,$(b))))

# --- shared kernel library -------------------------------------------
# Every test links sw/kernel's libvortex.a. Build it once, before the
# per-test makes run in parallel, so they don't each start the same build.
.PHONY: kernel-lib
kernel-lib:
	$(MAKE) -C $(ROOT_DIR)/sw/kernel

# --- pattern recipes --------------------------------------------------
.PHONY: build-% clean-%

build-%: kernel-lib
	$(MAKE) -C $* all

clean-%:
	$(MAKE) -C $* clean
