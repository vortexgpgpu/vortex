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
VIVADO_SRCS_DIR := $(BUILD_DIR)/vivado_srcs
VIVADO_SRCS_LIST := $(BUILD_DIR)/vivado_srcs.f

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

# Only XLEN and the TCU/FPU selectors are passed. Every other scalar is computed
# by the generated VX_config.vh behind an ifndef, so defining them here replaces
# those computations with the generator's report values -- which sets the memory
# port counts to zero and yields a design with no memory interface at all.
TYPE_DEFINES := $(filter -DVX_CFG_TCU_TYPE_% -DVX_CFG_FPU_TYPE_%,$(CONFIGS))

# The same defines every other synthesis flow passes (hw/syn/xilinx, hw/syn/altera).
# VX_dp_ram and VX_sp_ram gate all block-RAM inference behind SYNTHESIS, and
# gen_sources.sh preprocesses the tree, so without it the BRAM branch is not
# merely inactive -- it is deleted before Vivado sees it, and every memory in the
# design lands in distributed RAM. A U55C has 600960 LUTs usable as memory and
# this design asks for 753744 of them without it.
SYNTH_DEFINES := -DSYNTHESIS -DVIVADO -DNDEBUG

# This wrapper exposes a single AXI memory port, so give Vortex a single
# platform memory bank to match. Left at the default of 2, Vortex builds two
# internal memory ports and VX_mem_to_axi arbitrates them down onto the one
# bank, recovering the originating port from the *returned* AXI id
# (rsp_xbar_sel_in = m_axi_rid[0 +: NUM_PORTS_IN_BITS]). That makes correctness
# depend on the memory model echoing rid bit-exact -- an assumption real AXI
# satisfies but a timing model need not. One port removes the dependency
# entirely and matches what the wrapper actually presents.
PLATFORM_DEFINES := -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=1

# Paired with idBits in VortexConfig (VortexTarget.scala) -- change neither alone.
# That one shapes the IO the blackbox is instantiated with, this one shapes the
# module's own ports, and Verilog pads or truncates a disagreement between them
# without a diagnostic.
#
# The XRT default is 32, which is a width rather than a count: on XRT it only has
# to be wide enough. FASED reads it as a count, and is bounded from both sides.
#
# Too wide and `1 << idBits` overflows Scala's Int: at 32 it wraps to 1, leaving
# one response queue shared by every AXI id, so responses come back carrying the
# requesting transaction's id and another transaction's data.
#
# Too wide and it also does not fit. The response deinterleaver and the id-yank
# queues carry per-id state -- at 8 that is 77 deinterleaver instances of 2Kx64
# distributed RAM plus 512 yank queues, 750955 LUTs as memory against the 600960
# a U55C has.
#
# 4 is what Vortex actually drives: VX_mem_to_axi's tag buffer is engaged here,
# since UUID_WIDTH puts the upstream tag past any AXI id width, so reads carry
# CLOG2(TAG_BUFFER_SIZE) = 4 bits and writes never have their id consumed. It
# also matches the model's maxReads of 16. Raising it needs the utilization
# checked, not just the overflow avoided.
PLATFORM_DEFINES += -DPLATFORM_MEMORY_ID_WIDTH=4

# Vortex's performance counters are the only view inside the core that does not
# require re-elaborating the design: at a stall they say whether warps are still
# resident, whether they are waiting at a barrier, and which pipeline is holding
# them. The AXI boundary cannot show any of that -- a core waiting on a response
# it never retired looks exactly like a core with nothing left to do. They are
# read over the DCR response path, which both the harness and the driver already
# drive, but VX_csr_data.sv gates them behind PERF_ENABLE. Off by default because
# they cost area; build with PERF_DEFINES=-DPERF_ENABLE to read them, and note
# that without it every counter reads zero, which is indistinguishable from an
# idle core rather than an absent instrument.
PERF_DEFINES ?=
PLATFORM_DEFINES += $(PERF_DEFINES)

GEN_CFLAGS := -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN) $(TYPE_DEFINES) $(PLATFORM_DEFINES) $(SYNTH_DEFINES) $(RTL_INCLUDE)

# Only the packages and the wrapper are named; every other Vortex module is
# resolved by library search over RTL_INCLUDE. Handing the whole tree over as a
# flat list instead makes Verilator stop treating the macro-computed width
# parameters as constants.
# Packages have to be parsed before the modules that import them, so they are
# listed rather than left to library search, in dependency order. Vortex.sv
# imports gpu, trace and tlb; fpu is pulled in below the core.
EXPLICIT_NAMES := VX_gpu_pkg.sv VX_fpu_pkg.sv VX_tlb_pkg.sv VX_trace_pkg.sv \
                  VX_firesim_wrap.sv
EXPLICIT_SRCS := $(addprefix $(VIVADO_SRCS_DIR)/,$(EXPLICIT_NAMES))

sources: $(SRCS_LIST)
$(SRCS_LIST): $(VIVADO_SRCS_LIST)
	@mkdir -p $(BUILD_DIR)
	@printf '%s\n' $(EXPLICIT_SRCS) > $@
	@echo "wrote $@ ($(words $(EXPLICIT_SRCS)) explicit sources; rest via library search)"

# The include paths the simulator build must hand to Verilator, emitted so the
# build scripts do not have to restate them.
incdirs:
	@echo '$(RTL_INCLUDE)'

# Synthesis needs the opposite of simulation. Verilator resolves Vortex by
# library search, so the generated wrapper names only the packages; Vivado has no
# such search and must be handed every file. This flattens the tree into a
# self-contained directory with the macros inlined, which synthesis accepts --
# the constant-parameter failure that ruled flattening out for Verilator is an
# elaboration behaviour, not a synthesis one.

# The XRT AFU is a second kernel top with its own control shell. It arrives here
# only because its directory is on the include path for the shared port macros,
# and leaving it in gives Vivado two candidate tops, which makes it elaborate
# modules standalone and reject their interface ports. The header stays; the
# modules go.
XRT_AFU_MODULES := vortex_afu.v VX_afu_ctrl.sv VX_afu_wrap.sv

# Every input the flattener reads. Without these the rule below has no
# prerequisites at all, so make runs it exactly once and then reports "nothing to
# be done" forever: an edit under hw/rtl never reaches the flattened tree and the
# build silently consumes the previous elaboration. The generated headers are
# listed too because GEN_CFLAGS resolves widths out of them.
RTL_SRCS := $(shell find $(RTL_DIR) -name '*.sv' -o -name '*.v' -o -name '*.vh') \
            $(wildcard $(ROOT_DIR)/hw/VX_config.vh) $(wildcard $(ROOT_DIR)/hw/VX_types.vh) \
            $(firstword $(MAKEFILE_LIST))

vivado-sources: $(VIVADO_SRCS_LIST)
$(VIVADO_SRCS_LIST): $(RTL_SRCS)
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(SCRIPT_DIR)/gen_sources.sh -P $(GEN_CFLAGS) \
		-TVX_firesim_wrap -Cvivado_srcs -Ovivado_srcs.f
	@for f in $(XRT_AFU_MODULES); do \
		rm -f $(VIVADO_SRCS_DIR)/$$f; \
		grep -v "/$$f$$" $(VIVADO_SRCS_LIST) > $(VIVADO_SRCS_LIST).tmp && \
			mv $(VIVADO_SRCS_LIST).tmp $(VIVADO_SRCS_LIST); \
	done
	@# A design with no memory ports elaborates as zero-sized arrays deep inside
	@# Vivado, hours from here. Defining the config scalars instead of letting
	@# VX_config.vh compute them is enough to cause it, so check the result.
	@if grep -qE 'localparam +VX_MEM_PORTS *= *0 *;' $(VIVADO_SRCS_DIR)/VX_gpu_pkg.sv; then \
		echo "ERROR: generated sources resolve VX_MEM_PORTS to 0 (no memory interface)"; \
		echo "       pass only XLEN and the TCU/FPU selectors; VX_config.vh computes the rest"; \
		exit 1; \
	fi
	@echo "wrote $(VIVADO_SRCS_LIST) ($$(grep -cv '^+' $(VIVADO_SRCS_LIST)) files)"

check-firesim:
	@test -d $(FIRESIM_DIR)/sim || { \
		echo "FireSim checkout not found at $(FIRESIM_DIR)"; \
		echo "set FIRESIM_DIR to a FireSim 1.21.0 checkout"; \
		exit 1; }

clean-staged:
	@for f in $(SCALA_SRCS); do \
		rm -f $(TARGET_DEST)/$$(basename "$$f"); \
	done
