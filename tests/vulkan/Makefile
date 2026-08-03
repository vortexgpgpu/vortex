# Copyright © 2026  Vortex GPGPU
# SPDX-License-Identifier: MIT
#
# Vulkan test suite -- exercises the vortexpipe Gallium driver
# through Mesa's lavapipe ICD.

ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# --- master list ------------------------------------------------------
TESTS := compute cflow shmem triangle indexed tristrip cull ubo depth multidraw swom swom_draw3d textured draw3d raytrace rtquery rtquery_id \
         ssbo instanced mrt ubo_multiset rtquery_anyhit \
         tex_fetch tex_gather tex_shadow tex_lod tex_array tex_cube tex_3d tex_1d tex_border \
         stencil logicop blend

# --- common exclude list ---------------------------------------------
# Unsupported on the vortexpipe/simx path:
#   instanced, mrt, ubo_multiset : render hang
#   rtquery_anyhit               : any-hit does not cull non-opaque candidates
EXCLUDE := instanced mrt ubo_multiset rtquery_anyhit

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
