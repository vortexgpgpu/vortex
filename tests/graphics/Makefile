ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# Graphics extension tests (TEX / OM / RASTER blocks + full draw3d pipeline).
# Each subdir self-enables the EXT it exercises via its own CONFIGS guard,
# so `build-<test>` produces a correctly-configured kernel without extra
# flags. These are normally driven by ci/regression.sh's graphics() through
# ci/blackbox.sh (per-image args + PNG refs); this aggregator covers the
# build/clean sweep and manual use.
TESTS := gfx_tex gfx_om gfx_raster gfx_draw3d

BACKENDS := simx rtlsim opae xrt

.PHONY: all clean $(addprefix run-,$(BACKENDS))

all:   $(addprefix build-,$(TESTS))
clean: $(addprefix clean-,$(TESTS))

define backend_rule
.PHONY: run-$(1)
run-$(1): $$(addprefix run-$(1)-,$$(TESTS))
endef
$(foreach b,$(BACKENDS),$(eval $(call backend_rule,$(b))))

.PHONY: build-% clean-%

build-%:
	$(MAKE) -C $* all

clean-%:
	$(MAKE) -C $* clean

# run-<backend>-<test>: split "<backend>-<test>" on the first dash.
# Subdir names must not contain '-' (none currently do).
run-%:
	$(MAKE) -C $(word 2,$(subst -, ,$*)) run-$(word 1,$(subst -, ,$*))
