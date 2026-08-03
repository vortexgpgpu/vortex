# FireSim platform backend.
#
# The Chisel targets and the upstream patches live here, in the Vortex tree, so a
# fresh checkout can reproduce a build. The FireSim checkout itself is a working
# directory, not a source of truth: `stage` copies the targets into it and applies
# the patches, and anything left only in the checkout is expected to be lost.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

# configure copies this Makefile into the build tree, so ROOT_DIR resolves there
# while sources are read from VORTEX_HOME.

# The FireSim checkout to build in. One checkout is shared by every build tree,
# so this hangs off VORTEX_HOME rather than ROOT_DIR. Overridable so a scratch
# clone can be used without disturbing an existing one.
FIRESIM_DIR ?= $(VORTEX_HOME)/build_firesim/firesim

SRC_DIR := $(VORTEX_HOME)/hw/syn/firesim
PATCH_DIR := $(SRC_DIR)/patches

# Targets are compiled as part of FireSim's midasexamples project, which is where
# its harness and bridge helpers are resolvable from.
TARGET_DEST := $(FIRESIM_DIR)/sim/src/main/scala/midasexamples

SCALA_SRCS := $(wildcard $(SRC_DIR)/src/main/scala/*.scala)
CC_SRCS := $(wildcard $(SRC_DIR)/src/main/cc/*.cc)

# Test harnesses compile alongside FireSim's own midasexamples drivers.
CC_DEST := $(FIRESIM_DIR)/sim/src/main/cc/midasexamples

RTL_DIR := $(VORTEX_HOME)/hw/rtl
SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts

BUILD_DIR ?= $(ROOT_DIR)/hw/syn/firesim

# Flattened Vortex sources for the BlackBox. Golden Gate takes a file list, not
# include paths, so the tree is copied into one directory with the macros already
# inlined (-P) and emitted in elaboration order: packages, interfaces, modules.
SRCS_DIR := $(BUILD_DIR)/vortex_srcs
SRCS_LIST := $(BUILD_DIR)/vortex_srcs.f

.PHONY: stage patch sources check-firesim clean-staged

stage: check-firesim
	@mkdir -p $(TARGET_DEST) $(CC_DEST)
	@for f in $(SCALA_SRCS); do \
		cp -f "$$f" $(TARGET_DEST)/ || exit 1; \
		echo "staged $$(basename $$f)"; \
	done
	@for f in $(CC_SRCS); do \
		cp -f "$$f" $(CC_DEST)/ || exit 1; \
		echo "staged $$(basename $$f)"; \
	done

# Applies in series against a pristine checkout. Ordering matters: 0003 and 0005
# both build on 0001. Re-running against an already-patched checkout will report
# reversed hunks; that is the intended signal, not a failure to hide.
patch: check-firesim
	@cd $(FIRESIM_DIR) && for p in $(sort $(wildcard $(PATCH_DIR)/[0-9]*.patch)); do \
		echo "applying $$(basename $$p)"; \
		patch -p2 --no-backup-if-mismatch < "$$p" || exit 1; \
	done

XLEN ?= 32
CONFIGS ?= $(shell python3 $(VORTEX_HOME)/ci/gen_config.py --config=$(VORTEX_HOME)/VX_config.toml --cflags='-DVX_CFG_XLEN=$(XLEN)')

# VX_config.vh and VX_types.vh are generated into the build tree, so those come
# from ROOT_DIR while the RTL itself comes from VORTEX_HOME.
RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs \
               -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm \
               -I$(RTL_DIR)/cache -I$(RTL_DIR)/fpu -I$(RTL_DIR)/afu/xrt -I$(RTL_DIR)/afu/firesim

GEN_CFLAGS := -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN) $(CONFIGS) $(RTL_INCLUDE)

sources: $(SRCS_LIST)
$(SRCS_LIST):
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(SCRIPT_DIR)/gen_sources.sh -P $(GEN_CFLAGS) \
		-TVX_firesim_wrap -Cvortex_srcs -Ovortex_srcs.f

check-firesim:
	@test -d $(FIRESIM_DIR)/sim || { \
		echo "FireSim checkout not found at $(FIRESIM_DIR)"; \
		echo "set FIRESIM_DIR to a FireSim 1.21.0 checkout"; \
		exit 1; }

clean-staged:
	@for f in $(SCALA_SRCS); do \
		rm -f $(TARGET_DEST)/$$(basename "$$f"); \
	done
