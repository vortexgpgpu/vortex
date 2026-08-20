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
         stencil logicop blend dsclear firstvertex \
         alu fp16 subgroup atomic atomic_shared

# --- common exclude list ---------------------------------------------
# Unsupported on the vortexpipe/simx path:
#   (none)
#
# An entry here is a claim about today's driver, not a property of the test, so
# it has to be re-checked rather than inherited. instanced, mrt and ubo_multiset
# sat here for a month under "render hang" after the hang had been fixed, which
# cost three features their only coverage. Re-run an excluded test before
# trusting its entry.
EXCLUDE :=

# --- driver-defect coverage ------------------------------------------
# Each of these pins a specific defect in the vortexpipe driver and fails when
# it regresses; they are a separate assignment only so the master list above
# stays one block to edit.
TESTS += redraw realloc fallback_mix subgroup_vs bigwg manyvary image bgra bgra_om omfmt mrtfmt srgbblend flatvary subgroup_fs zeroshared deriv discard dispatchq imagedraw rtqdraw localarr

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
