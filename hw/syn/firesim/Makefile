# FireSim image build for the Xilinx cards, driven exactly as the native flows:
#
#   make PLATFORM=<xsa> [TARGET=hw|hw_emu]   an XRT card, as hw/syn/xilinx/xrt
#   make [DEVICE=<part>] [TARGET=hw|sim]     an aved card, as hw/syn/xilinx/aved
#
# with PREFIX, XLEN, CONFIGS and PERF=1 as there too. The image lands in
# $(BUILD_DIR)/bin as firesim.xclbin or firesim.vbin, beside firesim.card, which
# names the card for run-firesim; the runtime's xrtsim/avedsim targets need no
# image at all.
#
# FireSim itself is the vortexgpgpu/firesim fork, consumed through FIRESIM_PATH
# and pinned by FIRESIM_REV in VERSION: every change Vortex needs in FireSim is a
# commit there, and the checkout must contain the pinned one. The Chisel targets,
# harnesses and the card's platform config live here and are staged into it. The
# card's descriptor (hw/syn/xilinx/platforms/<card>.toml) supplies the FireSim
# platform, the FPGA part and device, the host-memory binding and its base.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk
include $(VORTEX_HOME)/hw/syn/common.mk

# configure copies this Makefile into the build tree, so ROOT_DIR resolves there
# while sources are read from VORTEX_HOME.

SRC_DIR := $(VORTEX_HOME)/hw/syn/firesim

# Targets are compiled as part of FireSim's midasexamples project, which is where
# its harness and bridge helpers are resolvable from.
TARGET_DEST := $(FIRESIM_PATH)/sim/src/main/scala/midasexamples

SCALA_SRCS := $(wildcard $(SRC_DIR)/src/main/scala/*.scala)
CC_SRCS := $(wildcard $(SRC_DIR)/src/main/cc/*.cc)

# Test harnesses compile alongside FireSim's own midasexamples drivers.
CC_DEST := $(FIRESIM_PATH)/sim/src/main/cc/midasexamples

RTL_DIR := $(VORTEX_HOME)/hw/rtl
SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts

XLEN ?= 32
PREFIX ?= build$(XLEN)

# Nothing given here on the command line reaches FireSim's own makes. A
# command-line variable propagates to every sub-make, and FireSim uses several
# of these names for its own purposes: PLATFORM is its platform (vitis/aved),
# not an XSA; TARGET reached v++ as `--target sim`; BUILD_DIR locates its
# elaborator jar. Each sub-make below is handed exactly what it needs instead.
MAKEOVERRIDES :=

# The card, named as the native flow for its kind names it. The runtime has an
# internal need to point the flattening elsewhere (FIRESIM_CARD, FIRESIM_WORK_DIR);
# those are not user knobs.
ifneq ($(FIRESIM_CARD),)
PLATFORM_GEN_ARGS := --card=$(FIRESIM_CARD)
else ifneq ($(PLATFORM),)
XSA := $(strip $(patsubst %.xpfm,%,$(notdir $(PLATFORM))))
DEV_ARCH := $(shell platforminfo -p $(PLATFORM) 2>/dev/null | grep 'FPGA Family' | sed 's/.*://' | sed '/ai_engine/d' | sed 's/^[[:space:]]*//')
PLATFORM_GEN_ARGS := --flow=xrt --xsa=$(XSA) --family='$(DEV_ARCH)'
else
DEVICE ?= xcv80-lsva4737-2MHP-e-S
PLATFORM_GEN_ARGS := --flow=aved --device=$(DEVICE)
endif
PLATFORM_GEN := python3 $(SCRIPT_DIR)/gen_platform.py $(PLATFORM_GEN_ARGS)

# The card is resolved before BUILD_DIR exists, because BUILD_DIR is named after
# it. $(shell) discards exit codes, so an empty answer means the generator
# failed, and it is asked again for its message.
FIRESIM_CARD := $(shell $(PLATFORM_GEN) --emit=card 2>/dev/null)
ifeq ($(FIRESIM_CARD),)
$(error $(shell $(PLATFORM_GEN) --emit=card 2>&1 >/dev/null))
endif
FIRESIM_PLATFORM := $(shell python3 $(SCRIPT_DIR)/gen_platform.py --card=$(FIRESIM_CARD) --emit=firesim-mk 2>&1 | sed -n 's/^FIRESIM_PLATFORM := //p')
ifeq ($(FIRESIM_PLATFORM),)
$(error $(shell python3 $(SCRIPT_DIR)/gen_platform.py --card=$(FIRESIM_CARD) --emit=firesim-mk 2>&1 >/dev/null))
endif

# The image each card can take, as its native flow names it.
TARGET ?= hw
ifeq ($(FIRESIM_PLATFORM), aved)
VALID_TARGETS := hw sim
else
VALID_TARGETS := hw hw_emu
endif
ifeq ($(FIRESIM_WORK_DIR),)
ifeq ($(filter $(TARGET),$(VALID_TARGETS)),)
$(error TARGET=$(TARGET) is not an image target for card $(FIRESIM_CARD): use one of $(VALID_TARGETS) \
        (xrtsim and avedsim are run targets and need no image))
endif
endif

# Named as the native flows name theirs, so FPGA_BIN_DIR means the same thing.
ifeq ($(FIRESIM_PLATFORM), aved)
BUILD_DIR := $(abspath $(PREFIX)_aved_$(TARGET))
else
BUILD_DIR := $(abspath $(PREFIX)_$(if $(XSA),$(XSA),$(FIRESIM_CARD))_$(TARGET))
endif
# The runtime's metasimulator reuses the simulation flattening from here, in a
# directory of its own.
ifneq ($(FIRESIM_WORK_DIR),)
BUILD_DIR := $(abspath $(FIRESIM_WORK_DIR))
endif
BIN_DIR := $(BUILD_DIR)/bin

FIRESIM_MK := $(BUILD_DIR)/firesim_platform.mk
$(shell mkdir -p $(BUILD_DIR) && $(PLATFORM_GEN) --emit=firesim-mk -o $(FIRESIM_MK))
include $(FIRESIM_MK)

# A user-named XSA is the one to build for; the descriptor's is the default.
ifneq ($(XSA),)
DEVICE := $(XSA)
endif

VORTEX_TARGET_CONFIG := VortexConfig

# The FireSim flow's make variables for this card's build.
FIRESIM_TARGET_ARGS := PLATFORM=$(FIRESIM_PLATFORM) TARGET_PROJECT=midasexamples \
                       DESIGN=VortexTarget TARGET_CONFIG=$(VORTEX_TARGET_CONFIG) PLATFORM_CONFIG=$(PLATFORM_CONFIG)
QUINTUPLET := $(FIRESIM_PLATFORM)-midasexamples-VortexTarget-$(VORTEX_TARGET_CONFIG)-$(PLATFORM_CONFIG)

# Vortex for the BlackBox. Golden Gate takes a file list, not include paths, so
# the tree is copied into one directory with the macros already inlined (-P) and
# emitted in elaboration order: packages, interfaces, modules. CIRCT then
# elaborates it into the one netlist the image reads, as every synthesis flow
# does (circt_elaborate.sh); the list names that netlist, and the RTL list stays
# next to it as $(VIVADO_SRCS_NAME).rtl.f.
#
# A simulation image (TARGET=sim or hw_emu) gets its own flattening, under
# -DSIMULATION, as the native xrt flow does for hw_emu.
ifneq ($(filter $(TARGET),sim hw_emu),)
IMAGE_KIND := sim
IMAGE_SUFFIX := _sim
endif
VIVADO_SRCS_NAME := vivado_srcs$(IMAGE_SUFFIX)
VIVADO_SRCS_DIR := $(BUILD_DIR)/$(VIVADO_SRCS_NAME)
VIVADO_SRCS_LIST := $(BUILD_DIR)/$(VIVADO_SRCS_NAME).f

# The same tree flattened for simulation instead. gen_sources.sh writes a
# directory, not a stamp, so the rule records one to hang the dependency on.
SRCS_DIR := $(BUILD_DIR)/vortex_srcs
SRCS_LIST := $(BUILD_DIR)/vortex_srcs.f
SIM_SRCS_STAMP := $(SRCS_DIR)/.sources.stamp

.PHONY: all help clean elaborate stage sources sim-sources vivado-sources xilinx-ips kernel-env image \
        metasim run-metasim check-firesim clean-staged

.DEFAULT_GOAL := all

help:
	@echo "FireSim image build (same tokens as hw/syn/xilinx/xrt and hw/syn/xilinx/aved):"
	@echo "  make PLATFORM=<xsa> [TARGET=hw|hw_emu]   XRT card"
	@echo "  make [DEVICE=<part>] [TARGET=hw|sim]     aved card (default: the V80)"
	@echo "  options: PREFIX=<dir prefix> XLEN=32|64 CONFIGS=\"-DVX_CFG_...\" PERF=1"
	@echo "  image:   <PREFIX>_<xsa|aved>_<TARGET>/bin/firesim.{xclbin,vbin}"
	@echo "  run:     make -C tests/regression/<app> run-firesim TARGET=<target> FPGA_BIN_DIR=<that bin>"
	@echo "           (TARGET=xrtsim|avedsim need no image)"
	@echo "  developer goals: metasim run-metasim sources clean"

# Copies only what changed: FireSim's make judges its elaboration by these
# files' timestamps, so rewriting identical files would redo Golden Gate (and
# relink the metasimulator) on every run.
stage: check-firesim
	@mkdir -p $(TARGET_DEST) $(CC_DEST)
	@for f in $(SCALA_SRCS); do \
		cmp -s "$$f" $(TARGET_DEST)/$$(basename $$f) || { cp -f "$$f" $(TARGET_DEST)/ && echo "staged $$(basename $$f)"; } || exit 1; \
	done
	@for f in $(CC_SRCS); do \
		cmp -s "$$f" $(CC_DEST)/$$(basename $$f) || { cp -f "$$f" $(CC_DEST)/ && echo "staged $$(basename $$f)"; } || exit 1; \
	done
	@$(PLATFORM_GEN) --emit=scala -o $(BUILD_DIR)/$(PLATFORM_CONFIG).scala
	@cmp -s $(BUILD_DIR)/$(PLATFORM_CONFIG).scala $(TARGET_DEST)/$(PLATFORM_CONFIG).scala || { \
		cp -f $(BUILD_DIR)/$(PLATFORM_CONFIG).scala $(TARGET_DEST)/ && \
		echo "staged $(PLATFORM_CONFIG).scala ($(FIRESIM_CARD))"; }

# Architecture overrides, as the other synthesis flows take them, e.g.
# CONFIGS="-DVX_CFG_NUM_WARPS=16 -DVX_CFG_EXT_TEX_ENABLE". XCONFIGS is the fully
# resolved set, used only to select extension sources and type selectors.
CONFIGS ?=
XCONFIGS := $(shell python3 $(VORTEX_HOME)/ci/gen_config.py --config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS) -DVX_CFG_XLEN=$(XLEN)')

# VX_config.vh and VX_types.vh are generated into the build tree, so those come
# from ROOT_DIR while the RTL itself comes from VORTEX_HOME.
RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs \
               -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm \
               -I$(RTL_DIR)/cache -I$(RTL_DIR)/fpu -I$(RTL_DIR)/afu/common -I$(RTL_DIR)/afu/firesim

RTL_PKGS :=
include $(VORTEX_HOME)/hw/syn/extensions.mk

# Only the overrides and the TCU/FPU selectors are passed. Every other scalar is
# computed by the generated VX_config.vh behind an ifndef, so passing the whole
# resolved XCONFIGS here replaces those computations with the generator's report
# values -- which sets the memory port counts to zero and yields a design with no
# memory interface at all.
TYPE_DEFINES := $(filter -DVX_CFG_TCU_TYPE_% -DVX_CFG_FPU_TYPE_%,$(XCONFIGS))

# The same defines every other synthesis flow passes (hw/syn/xilinx, hw/syn/altera).
# VX_dp_ram and VX_sp_ram gate all block-RAM inference behind SYNTHESIS, and
# gen_sources.sh preprocesses the tree, so without it the BRAM branch is not
# merely inactive -- it is deleted before Vivado sees it, and every memory in the
# design lands in distributed RAM. A U55C has 600960 LUTs usable as memory and
# this design asks for 753744 of them without it.
SYNTH_DEFINES := -DSYNTHESIS -DVIVADO -DNDEBUG
ifeq ($(IMAGE_KIND), sim)
SYNTH_DEFINES += -DSIMULATION
endif

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

# On aved the simulator's clock gates are turned into clock enables by
# synthesis (the V80's reconfigurable partition has no BUFGCE, and a fabric gate
# is not safe; see FireSimFabricClockGate). That conversion cannot enter a
# vendor core: each is synthesized out of context behind a kept hierarchy, so
# it would stay on a logic-driven clock and the platform build rejects it. So
# no unit may take one here: the FPU is the STD one, as in the metasimulator,
# its units' copies in the flattened tree take their soft path (VENDOR_FP_UNITS
# below), and the RTU uses the soft FMA and divider, with the soft divider's
# latency in place of xil_fdiv's.
ifeq ($(FIRESIM_PLATFORM), aved)
PLATFORM_DEFINES += -DVX_CFG_FPU_TYPE_STD -DVX_CFG_RTU_USE_DSP=0 -DVX_CFG_RTU_FDIV_LATENCY=17
VENDOR_FP_UNITS := VX_fma_unit.sv VX_fdiv_unit.sv VX_fsqrt_unit.sv
endif

# Vortex's performance counters are the only view inside the core that does not
# require re-elaborating the design: at a stall they say whether warps are still
# resident, whether they are waiting at a barrier, and which pipeline is holding
# them. The AXI boundary cannot show any of that -- a core waiting on a response
# it never retired looks exactly like a core with nothing left to do. They are
# read over the DCR response path, which both the harness and the driver already
# drive, but VX_csr_data.sv gates them behind PERF_ENABLE. Off by default because
# they cost area; build with PERF=1 to read them, as in the other synthesis
# flows, and note that without it every counter reads zero, which is
# indistinguishable from an idle core rather than an absent instrument.
ifdef PERF
PLATFORM_DEFINES += -DPERF_ENABLE
endif

# Synthesis resolves the unit types as a synthesis build: TYPE_DEFINES comes from
# a resolution without SYNTHESIS, which picks the simulation FPU (STD) over the
# FPGA one (DSP) and would hand synthesis a different design than Vortex's.
SYNTH_TYPE_DEFINES := $(filter -DVX_CFG_TCU_TYPE_% -DVX_CFG_FPU_TYPE_%,$(shell python3 $(VORTEX_HOME)/ci/gen_config.py \
	--config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS) -DVX_CFG_XLEN=$(XLEN) $(SYNTH_DEFINES) $(PLATFORM_DEFINES)'))
GEN_CFLAGS := -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN) $(CONFIGS) $(SYNTH_TYPE_DEFINES) $(PLATFORM_DEFINES) $(SYNTH_DEFINES) $(RTL_INCLUDE)

# Defines arrive on the command line, which no file timestamp reflects, so a
# changed CONFIGS would otherwise reuse the previous flattened tree. The stamp is
# rewritten only when the flags differ, which keeps an unchanged build current.
CONFIG_STAMP := $(BUILD_DIR)/gen_cflags$(IMAGE_SUFFIX).stamp
$(shell mkdir -p $(BUILD_DIR) && echo '$(GEN_CFLAGS)' > $(CONFIG_STAMP).tmp && \
        (cmp -s $(CONFIG_STAMP).tmp $(CONFIG_STAMP) && rm -f $(CONFIG_STAMP).tmp || mv -f $(CONFIG_STAMP).tmp $(CONFIG_STAMP)))

# The metasimulator needs the opposite of both synthesis defines, so it gets its
# own flattening rather than reusing the Vivado one.
#
# -DVIVADO selects the Xilinx floating_point cores (xil_fma, xil_fdiv) over the
# soft FPU. Those are generated .xci IP with no Verilog to elaborate, so a
# metasim built from Vivado-flattened sources fails on the missing modules.
#
# -DSYNTHESIS gates VX_dp_ram/VX_sp_ram onto their BRAM-inference branch; the
# behavioural models are what a simulator wants, and they carry the read/write
# collision semantics the inference branch leaves to the tool.
SIM_DEFINES := -DNDEBUG
SIM_GEN_CFLAGS := -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN) $(CONFIGS) $(TYPE_DEFINES) $(PLATFORM_DEFINES) $(SIM_DEFINES) $(RTL_INCLUDE)

SIM_CONFIG_STAMP := $(BUILD_DIR)/sim_cflags.stamp
$(shell mkdir -p $(BUILD_DIR) && echo '$(SIM_GEN_CFLAGS)' > $(SIM_CONFIG_STAMP).tmp && \
        (cmp -s $(SIM_CONFIG_STAMP).tmp $(SIM_CONFIG_STAMP) && rm -f $(SIM_CONFIG_STAMP).tmp || mv -f $(SIM_CONFIG_STAMP).tmp $(SIM_CONFIG_STAMP)))

# Only the packages and the wrapper are named; every other Vortex module is
# resolved by library search over RTL_INCLUDE. Handing the whole tree over as a
# flat list instead makes Verilator stop treating the macro-computed width
# parameters as constants.
# Packages have to be parsed before the modules that import them, so they are
# listed rather than left to library search, in dependency order. Vortex.sv
# imports gpu, trace and tlb; fpu is pulled in below the core.
# Extension packages follow in dependency order, which is not extensions.mk's
# append order: raster imports om.
EXT_PKG_ORDER := VX_tcu_pkg.sv VX_dxa_pkg.sv VX_rtu_pkg.sv VX_om_pkg.sv \
                 VX_raster_pkg.sv VX_tex_pkg.sv
EXT_PKG_NAMES := $(filter $(notdir $(RTL_PKGS)),$(EXT_PKG_ORDER))
PKG_ORDER := VX_gpu_pkg.sv VX_fpu_pkg.sv VX_tlb_pkg.sv VX_trace_pkg.sv $(EXT_PKG_NAMES)
EXPLICIT_NAMES := $(PKG_ORDER) VX_firesim_wrap.sv
EXPLICIT_SRCS := $(addprefix $(SRCS_DIR)/,$(EXPLICIT_NAMES))
empty :=
space := $(empty) $(empty)

sources: $(SRCS_LIST)
$(SRCS_LIST): $(SIM_SRCS_STAMP)
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

# The shared AFU shell (VX_afu_wrap and its control, demux, drain and reset
# modules) is a second kernel top. It arrives here only because its directory is
# on the include path for the shared port macros in vortex_afu.vh, and leaving it
# in gives Vivado two candidate tops, which makes it elaborate modules standalone
# and reject their interface ports. The header stays; the modules go.
AFU_SHELL_MODULES := $(notdir $(wildcard $(RTL_DIR)/afu/common/*.sv))

# Every input the flattener reads. Without these the rule below has no
# prerequisites at all, so make runs it exactly once and then reports "nothing to
# be done" forever: an edit under hw/rtl never reaches the flattened tree and the
# build silently consumes the previous elaboration. The generated headers are
# listed too because GEN_CFLAGS resolves widths out of them.
RTL_SRCS := $(shell find $(RTL_DIR) -name '*.sv' -o -name '*.v' -o -name '*.vh') \
            $(wildcard $(ROOT_DIR)/hw/VX_config.vh) $(wildcard $(ROOT_DIR)/hw/VX_types.vh) \
            $(firstword $(MAKEFILE_LIST)) $(CONFIG_STAMP)

vivado-sources: $(VIVADO_SRCS_LIST)
$(VIVADO_SRCS_LIST): $(RTL_SRCS)
	@mkdir -p $(BUILD_DIR)
	@# gen_sources.sh adds to its directory without clearing it, so a source the
	@# previous configuration needed (a TEX file, say) would stay and be listed.
	rm -rf $(VIVADO_SRCS_DIR)
	cd $(BUILD_DIR) && $(SCRIPT_DIR)/gen_sources.sh -P $(GEN_CFLAGS) \
		-TVX_firesim_wrap -C$(VIVADO_SRCS_NAME) -O$(VIVADO_SRCS_NAME).f
	@for f in $(AFU_SHELL_MODULES); do \
		rm -f $(VIVADO_SRCS_DIR)/$$f; \
		grep -v "/$$f$$" $(VIVADO_SRCS_LIST) > $(VIVADO_SRCS_LIST).tmp && \
			mv $(VIVADO_SRCS_LIST).tmp $(VIVADO_SRCS_LIST); \
	done
	@for f in $(VENDOR_FP_UNITS); do \
		[ -f $(VIVADO_SRCS_DIR)/$$f ] || continue; \
		grep -q 'localparam VENDOR_OK = 1;' $(VIVADO_SRCS_DIR)/$$f || \
			{ echo "ERROR: $$f no longer selects its vendor core by VENDOR_OK"; exit 1; }; \
		sed -i 's/localparam VENDOR_OK = 1;/localparam VENDOR_OK = 0;/' $(VIVADO_SRCS_DIR)/$$f; \
	done
	@# gen_sources.sh lists packages in find(1) order, which is directory order,
	@# not dependency order. Synthesis reorders on its own; a simulator compiles
	@# the list as given. Under -DSIMULATION VX_trace_pkg imports VX_gpu_pkg, and
	@# with trace first xsim failed on "'VX_gpu_pkg' is not declared". Put the
	@# packages in the dependency order the metasim list already uses.
	@{ grep '^+' $(VIVADO_SRCS_LIST); \
	   for p in $(PKG_ORDER); do grep "/$$p$$" $(VIVADO_SRCS_LIST); done; \
	   grep -v '^+' $(VIVADO_SRCS_LIST) | grep -vE '/($(subst $(space),|,$(PKG_ORDER)))$$'; \
	 } > $(VIVADO_SRCS_LIST).tmp && mv $(VIVADO_SRCS_LIST).tmp $(VIVADO_SRCS_LIST)
	@# A design with no memory ports elaborates as zero-sized arrays deep inside
	@# Vivado, hours from here. Defining the config scalars instead of letting
	@# VX_config.vh compute them is enough to cause it, so check the result.
	@if grep -qE 'localparam +VX_MEM_PORTS *= *0 *;' $(VIVADO_SRCS_DIR)/VX_gpu_pkg.sv; then \
		echo "ERROR: generated sources resolve VX_MEM_PORTS to 0 (no memory interface)"; \
		echo "       pass only XLEN and the TCU/FPU selectors; VX_config.vh computes the rest"; \
		exit 1; \
	fi
	mv $(VIVADO_SRCS_LIST) $(BUILD_DIR)/$(VIVADO_SRCS_NAME).rtl.f
	$(SCRIPT_DIR)/circt_elaborate.sh -tVX_firesim_wrap -l$(BUILD_DIR)/$(VIVADO_SRCS_NAME).rtl.f \
		-x$(SCRIPT_DIR)/xilinx_ip_stubs.sv -o$(BUILD_DIR)/$(VIVADO_SRCS_NAME).elab.sv
	realpath $(BUILD_DIR)/$(VIVADO_SRCS_NAME).elab.sv > $(VIVADO_SRCS_LIST)
	@echo "wrote $(VIVADO_SRCS_LIST): $(VIVADO_SRCS_NAME).elab.sv, elaborated from $$(grep -cv '^+' $(BUILD_DIR)/$(VIVADO_SRCS_NAME).rtl.f) files"

# The same flattening under the simulation defines. Verilator finds the modules
# by library search over this directory, so only the tree has to be complete --
# the file list the search is seeded with is $(SRCS_LIST) above.
SIM_RTL_SRCS := $(filter-out $(CONFIG_STAMP),$(RTL_SRCS)) $(SIM_CONFIG_STAMP)

sim-sources: $(SIM_SRCS_STAMP)
$(SIM_SRCS_STAMP): $(SIM_RTL_SRCS)
	@mkdir -p $(BUILD_DIR)
	@# Cleared for the same reason as the Vivado flattening above.
	rm -rf $(SRCS_DIR)
	cd $(BUILD_DIR) && $(SCRIPT_DIR)/gen_sources.sh -P $(SIM_GEN_CFLAGS) \
		-TVX_firesim_wrap -Cvortex_srcs -Ovortex_srcs_all.f
	@# The same zero-memory-port check the Vivado flattening makes: here it would
	@# surface as a zero-width AXI port rather than an hours-later elaboration
	@# failure, but it is the same defect and cheaper to catch at the source.
	@if grep -qE 'localparam +VX_MEM_PORTS *= *0 *;' $(SRCS_DIR)/VX_gpu_pkg.sv; then \
		echo "ERROR: generated sources resolve VX_MEM_PORTS to 0 (no memory interface)"; \
		echo "       pass only XLEN and the TCU/FPU selectors; VX_config.vh computes the rest"; \
		exit 1; \
	fi
	@# The vendor FP branch is a generate-if on a localparam, not an ifdef, so the
	@# instantiation survives preprocessing either way. What matters is that the
	@# localparam resolves to 0 so the branch is dead -- Verilator then prunes it,
	@# as the rtlsim flow has always relied on. Checking for the instantiation text
	@# instead reports every build as broken.
	@if grep -qE 'localparam +VENDOR_OK *= *1 *;' $(SRCS_DIR)/VX_fma_unit.sv; then \
		echo "ERROR: simulation sources select the Xilinx FP IP (VENDOR_OK=1)"; \
		echo "       SIM_DEFINES must not carry -DVIVADO or -DQUARTUS"; \
		exit 1; \
	fi
	touch $@
	@echo "wrote $(SRCS_DIR) ($$(ls $(SRCS_DIR) | wc -l) files)"

# Vendor floating-point cores. On Vivado the FPU and RTU select Xilinx's
# floating_point IP (xil_fma, xil_fdiv, ...) instead of the soft core, and the
# flattened sources only instantiate them, so the kernel packaging must be handed
# the generated cores or synthesis fails on the missing modules. Same script and
# core set as the XRT flow, for the card's part (FPGA_PART, from the descriptor).
XILINX_IP_DIR := $(BUILD_DIR)/ip
XILINX_IP_STAMP := $(XILINX_IP_DIR)/.part-$(FPGA_PART)

xilinx-ips: $(XILINX_IP_STAMP)
$(XILINX_IP_STAMP): $(SCRIPT_DIR)/xilinx_ip_gen.tcl
	rm -rf $(XILINX_IP_DIR)
	cd $(BUILD_DIR) && vivado -mode batch -nojournal -nolog -source $(SCRIPT_DIR)/xilinx_ip_gen.tcl \
		-tclargs $(XILINX_IP_DIR) $(FPGA_PART)
	@test -n "$$(find $(XILINX_IP_DIR) -name '*.xci')" || { echo "ERROR: no .xci generated under $(XILINX_IP_DIR)"; exit 1; }
	touch $@

# The environment the FireSim Vitis kernel build reads: the netlist and vendor
# cores package_kernel.tcl attaches.
kernel-env:
	@echo 'export VORTEX_VIVADO_SRCS=$(VIVADO_SRCS_LIST)'
	@echo 'export VORTEX_XILINX_IP_DIR=$(XILINX_IP_DIR)'

# Simulator clock (MHz): the image is timed at it and the shell programs it.
# V80: the Fmax measured on a FireSim image of Vortex (143.4 MHz).
ifeq ($(FIRESIM_PLATFORM), aved)
FREQUENCY ?= 143.0
else
FREQUENCY ?= 60.0
endif
STRATEGY ?= TIMING

# How the card's FireSim platform spells this TARGET, and where it leaves the
# image. aved builds both images from one goal and selects with its own TARGET,
# putting each in build_dir.$(TARGET); vitis has a separate `sim` goal for
# hw_emu, and must never see TARGET at all -- its Makefile sets
# `bitstream: TARGET = hw` as a target-specific variable, which a passed-in
# TARGET would beat, sending `--target sim` to v++.
PLATFORM_BUILD_DIR := $(FIRESIM_PATH)/platforms/$(FIRESIM_PLATFORM)/cl_$(QUINTUPLET)
ifeq ($(FIRESIM_PLATFORM), aved)
PLATFORM_IMAGE_GOAL := bitstream TARGET=$(TARGET)
PLATFORM_IMAGE := $(PLATFORM_BUILD_DIR)/bitstream/build_dir.$(TARGET)/firesim.vbin
IMAGE_NAME := firesim.vbin
else ifeq ($(TARGET), hw_emu)
# Without debug: a debug hw_emu kernel logs a waveform of the whole shell on
# every run (~800 MB for demo, tens of GB for raycast) and runs far slower.
PLATFORM_IMAGE_GOAL := sim VPP_DEBUG=
PLATFORM_IMAGE := $(PLATFORM_BUILD_DIR)/simulation/build_dir.$(DEVICE)/firesim.xclbin
IMAGE_NAME := firesim.xclbin
else
PLATFORM_IMAGE_GOAL := bitstream
PLATFORM_IMAGE := $(PLATFORM_BUILD_DIR)/bitstream/build_dir.$(DEVICE)/firesim.xclbin
IMAGE_NAME := firesim.xclbin
endif

# One command from sources to the card's image, collected into $(BIN_DIR) with
# what running it needs beside it. Expects the FireSim environment
# (ci/firesim_env.sh) and the Xilinx tools for the card's platform.
all: image
	@mkdir -p $(BIN_DIR)
	cp -f $(PLATFORM_IMAGE) $(BIN_DIR)/$(IMAGE_NAME)
	@echo $(FIRESIM_CARD) > $(BIN_DIR)/firesim.card
ifeq ($(TARGET), hw_emu)
	emconfigutil --platform $(DEVICE) --od $(BIN_DIR)
endif
	@echo "image: $(abspath $(BIN_DIR))/$(IMAGE_NAME) (card $(FIRESIM_CARD), TARGET=$(TARGET))"

# The FireSim platform's own build. $(SRCS_LIST) is deliberately not a
# prerequisite: the recipe hands synthesis the Vivado-flattened list, and pulling
# the simulation one in would flatten the tree a second time under the opposite
# defines for nothing.
image: vivado-sources xilinx-ips stage
	$(MAKE) -C $(FIRESIM_PATH)/sim replace-rtl $(FIRESIM_TARGET_ARGS)
	VORTEX_VIVADO_SRCS=$(VIVADO_SRCS_LIST) VORTEX_XILINX_IP_DIR=$(XILINX_IP_DIR) \
	$(MAKE) -C $(PLATFORM_BUILD_DIR) $(PLATFORM_IMAGE_GOAL) \
		DEVICE=$(DEVICE) HOST_MEM_SP='$(HOST_MEM_SP)' FREQUENCY=$(FREQUENCY) STRATEGY=$(STRATEGY)

# Golden Gate's elaboration of the card's target on its own, for the runtime's
# metasimulator (xrtsim/avedsim), which needs the generated RTL and headers but
# no image. An image build elaborates as part of replace-rtl.
elaborate: stage
	$(MAKE) -C $(FIRESIM_PATH)/sim verilog $(FIRESIM_TARGET_ARGS)

clean:
	rm -rf $(BUILD_DIR)

# The metasimulator: the FireSim-transformed design in Verilator, driven through
# the DPI emul simif. No FPGA and no Vivado, so it runs the same target the card
# runs without needing either -- which is what makes it the step before silicon.
#
# Vortex is supplied here rather than inlined into the generated wrapper (see
# VortexTarget.scala): the packages and the wrapper are named and everything else
# resolves by library search over the flattened tree. Naming the whole tree
# instead makes Verilator stop folding the macro-computed widths into constants.
#
# The list is collapsed onto one line because it reaches Verilator through a
# shell, which would otherwise split it on the newlines.
# hw/syn/common.mk exports VERILATOR as a bare path to Vortex's own toolchain,
# and FireSim declares `VERILATOR ?= verilator --main --timing --cc --exe`. A
# `?=` yields to the environment, so the inherited value silently strips every
# flag and Verilator exits asking for --cc. Drop it from the environment instead
# of restating FireSim's flags here, which would rot against its Makefrag. The
# metasim then builds with the Verilator on PATH, which FireSim's env.sh pins.
metasim: sources stage
	env -u VERILATOR $(MAKE) -C $(FIRESIM_PATH)/sim verilator $(FIRESIM_TARGET_ARGS) \
		EXTRA_VERILATOR_FLAGS="-Wno-fatal -y $(SRCS_DIR) +libext+.sv+.v $$(tr '\n' ' ' < $(SRCS_LIST))"

# Runs it. ARGS reaches the harness; the metasim needs no image, since the design
# is the simulator.
#
# Not dependent on `metasim`, so running never rebuilds the 460 MB model behind
# the caller's back. Build with `make metasim` first.
run-metasim:
	env -u VERILATOR $(MAKE) -C $(FIRESIM_PATH)/sim run-verilator $(FIRESIM_TARGET_ARGS) ARGS="$(ARGS)"

# The checkout must contain the pinned fork commit. Being ahead of it is normal
# while developing the fork; being behind it, or on unrelated history, builds
# against FireSim changes this tree does not expect.
check-firesim:
	@test -d $(FIRESIM_PATH)/sim || { \
		echo "FireSim checkout not found at $(FIRESIM_PATH)"; \
		echo "clone vortexgpgpu/firesim there (see docs/building_toolchain.md)"; \
		exit 1; }
	@git -C $(FIRESIM_PATH) merge-base --is-ancestor $(FIRESIM_REV) HEAD 2>/dev/null || { \
		echo "FireSim checkout at $(FIRESIM_PATH) does not contain FIRESIM_REV=$(FIRESIM_REV)"; \
		echo "  HEAD is $$(git -C $(FIRESIM_PATH) rev-parse --short HEAD)"; \
		echo "  fetch the fork and check out $(FIRESIM_REV) or a descendant"; \
		exit 1; }

clean-staged:
	@for f in $(SCALA_SRCS) $(PLATFORM_CONFIG).scala; do \
		rm -f $(TARGET_DEST)/$$(basename "$$f"); \
	done
