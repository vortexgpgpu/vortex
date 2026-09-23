# FireSim platform backend.
#
# FireSim itself is the vortexgpgpu/firesim fork, consumed through FIRESIM_PATH
# and pinned by FIRESIM_REV in VERSION: every change Vortex needs in FireSim is a
# commit there, and the checkout must contain the pinned one. The Chisel targets,
# harnesses and the card's platform config live here and are staged into it.
#
# The card comes from its descriptor (hw/syn/xilinx/platforms/<card>.toml):
# FIRESIM_CARD selects it, and the descriptor supplies the FireSim platform, the
# FPGA part and device, the host-memory binding and its base address.

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

BUILD_DIR ?= $(ROOT_DIR)/hw/syn/firesim

FIRESIM_CARD ?= u55c
PLATFORM_GEN := python3 $(SCRIPT_DIR)/gen_platform.py --card=$(FIRESIM_CARD)
FIRESIM_MK := $(BUILD_DIR)/firesim_platform.mk
FIRESIM_GEN_RC := $(shell mkdir -p $(BUILD_DIR) && $(PLATFORM_GEN) --emit=firesim-mk -o $(FIRESIM_MK) 2> $(FIRESIM_MK).err; echo $$?)
ifneq ($(FIRESIM_GEN_RC),0)
$(error $(shell cat $(FIRESIM_MK).err))
endif
include $(FIRESIM_MK)

# The FireSim flow's make variables for this card's build.
FIRESIM_TARGET_ARGS := PLATFORM=$(PLATFORM) TARGET_PROJECT=midasexamples \
                       DESIGN=VortexTarget TARGET_CONFIG=VortexConfig PLATFORM_CONFIG=$(PLATFORM_CONFIG)
QUINTUPLET := $(PLATFORM)-midasexamples-VortexTarget-VortexConfig-$(PLATFORM_CONFIG)

# Flattened Vortex sources for the BlackBox. Golden Gate takes a file list, not
# include paths, so the tree is copied into one directory with the macros already
# inlined (-P) and emitted in elaboration order: packages, interfaces, modules.
VIVADO_SRCS_DIR := $(BUILD_DIR)/vivado_srcs
VIVADO_SRCS_LIST := $(BUILD_DIR)/vivado_srcs.f

# The same tree flattened for simulation instead. gen_sources.sh writes a
# directory, not a stamp, so the rule records one to hang the dependency on.
SRCS_DIR := $(BUILD_DIR)/vortex_srcs
SRCS_LIST := $(BUILD_DIR)/vortex_srcs.f
SIM_SRCS_STAMP := $(SRCS_DIR)/.sources.stamp

.PHONY: stage sources sim-sources vivado-sources xilinx-ips kernel-env bitstream \
        metasim run-metasim check-firesim clean-staged

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
	@$(PLATFORM_GEN) --emit=scala -o $(TARGET_DEST)/$(PLATFORM_CONFIG).scala
	@echo "staged $(PLATFORM_CONFIG).scala ($(FIRESIM_CARD))"

XLEN ?= 32

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

GEN_CFLAGS := -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN) $(CONFIGS) $(TYPE_DEFINES) $(PLATFORM_DEFINES) $(SYNTH_DEFINES) $(RTL_INCLUDE)

# Defines arrive on the command line, which no file timestamp reflects, so a
# changed CONFIGS would otherwise reuse the previous flattened tree. The stamp is
# rewritten only when the flags differ, which keeps an unchanged build current.
CONFIG_STAMP := $(BUILD_DIR)/gen_cflags.stamp
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
EXPLICIT_NAMES := VX_gpu_pkg.sv VX_fpu_pkg.sv VX_tlb_pkg.sv VX_trace_pkg.sv \
                  $(EXT_PKG_NAMES) VX_firesim_wrap.sv
EXPLICIT_SRCS := $(addprefix $(SRCS_DIR)/,$(EXPLICIT_NAMES))

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
	cd $(BUILD_DIR) && $(SCRIPT_DIR)/gen_sources.sh -P $(GEN_CFLAGS) \
		-TVX_firesim_wrap -Cvivado_srcs -Ovivado_srcs.f
	@for f in $(AFU_SHELL_MODULES); do \
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

# The same flattening under the simulation defines. Verilator finds the modules
# by library search over this directory, so only the tree has to be complete --
# the file list the search is seeded with is $(SRCS_LIST) above.
SIM_RTL_SRCS := $(filter-out $(CONFIG_STAMP),$(RTL_SRCS)) $(SIM_CONFIG_STAMP)

sim-sources: $(SIM_SRCS_STAMP)
$(SIM_SRCS_STAMP): $(SIM_RTL_SRCS)
	@mkdir -p $(BUILD_DIR)
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

# The environment the FireSim Vitis kernel build reads: the sources and vendor
# cores package_kernel.tcl attaches, and the pre-opt hook the link runs to
# resolve the async-BRAM placeholders (it locates hw/scripts through TOOL_DIR).
kernel-env:
	@echo 'export VORTEX_VIVADO_SRCS=$(VIVADO_SRCS_LIST)'
	@echo 'export VORTEX_XILINX_IP_DIR=$(XILINX_IP_DIR)'
	@echo 'export VORTEX_PRE_OPT_TCL=$(SRC_DIR)/pre_opt_hook.tcl'
	@echo 'export TOOL_DIR=$(SCRIPT_DIR)'

# One command from sources to the card's image. Expects the FireSim environment
# (its env.sh) and the Xilinx tools for the card's platform in the environment.
# $(SRCS_LIST) is deliberately not a prerequisite: the recipe hands synthesis the
# Vivado-flattened list, and pulling the simulation one in would flatten the tree
# a second time under the opposite defines for nothing.
bitstream: vivado-sources xilinx-ips stage
	$(MAKE) -C $(FIRESIM_PATH)/sim replace-rtl $(FIRESIM_TARGET_ARGS)
	VORTEX_VIVADO_SRCS=$(VIVADO_SRCS_LIST) VORTEX_XILINX_IP_DIR=$(XILINX_IP_DIR) \
	VORTEX_PRE_OPT_TCL=$(SRC_DIR)/pre_opt_hook.tcl TOOL_DIR=$(SCRIPT_DIR) \
	$(MAKE) -C $(FIRESIM_PATH)/platforms/$(PLATFORM)/cl_$(QUINTUPLET) bitstream \
		DEVICE=$(DEVICE) HOST_MEM_SP='$(HOST_MEM_SP)' FREQUENCY=$(FREQUENCY) STRATEGY=$(STRATEGY)

FREQUENCY ?= 60.0
STRATEGY ?= TIMING

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
# Deliberately not dependent on `metasim`: `stage` copies the harness sources
# unconditionally, so every invocation would bump their mtimes and relink the
# 460 MB model before running it. Build with `make metasim` first.
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
