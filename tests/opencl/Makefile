ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

# --- master list ------------------------------------------------------
TESTS := \
    vecadd sgemm conv3 psort saxpy sfilter sgemm2 sgemm3 psum oclprintf \
    dotproduct transpose spmv stencil lbm nearn guassian kmeans         \
    blackscholes bfs copybuf histogram atomicreduce                     \
    pathfinder nw lud hotspot hotspot3D srad backprop streamcluster     \
    lavaMD btree cfd dwt2d myocyte heartwall hybridsort                  \
    image_copy image_linear image_int image_get \
    image_ff image_ff_linear image_ff_bgra image_hwsw

# --- common exclude list ---------------------------------------------
# histogram/atomicreduce/hybridsort use atomic_add (RVA amo*.w); run them only
# with the A extension (CONFIGS="-DVX_CFG_EXT_A_ENABLE"), so exclude them from
# the default sweep which builds for the no-atomics config.
#
# image_* need the OpenCL image feature, which requires an image-enabled
# PoCL-Vortex build (CL_DEVICE_IMAGE_SUPPORT + the vortex scalar image builtins).
# They are excluded from the default sweep and run via their own opencl.yaml
# block that pins the image-enabled POCL_PATH.
EXCLUDE := \
    transpose \
    lbm       \
    copybuf   \
    histogram \
    atomicreduce \
    hybridsort \
    image_copy \
    image_linear \
    image_int \
    image_get \
    image_ff \
    image_ff_linear \
    image_ff_bgra \
    image_hwsw

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
