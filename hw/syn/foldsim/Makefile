# foldsim image build for the V80, driven as the native aved flow
# (hw/syn/xilinx/aved):
#
#   make [DEVICE=<part>] [PREFIX=...] [CONFIGS="-DVX_CFG_..."] [PERF=1]
#        [FOLDSIM_CLASSES="..."] [FOLDSIM_LIFT="..."] [CIRCT_BIN=<dir>] [FOLDSIM=<driver>]
#
# foldsim compiles Vortex into a simulator, on CIRCT: each class threaded
# through one physical copy (FAME-5), the rest of the target a model of its
# own, and every channel between them in one token fabric. The kernel
# (hw/rtl/afu/foldsim) adds the memory and control endpoints around it. The
# image lands in $(BUILD_DIR)/bin/foldsim.vbin; run it with
#   FPGA_BIN_DIR=<that bin> ./ci/blackbox.sh --driver=foldsim --target=hw ...
# The runtime's avedsim target simulates the same generated design in-process.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk
include $(VORTEX_HOME)/hw/syn/common.mk

SRC_DIR := $(VORTEX_HOME)/hw/syn/foldsim
RTL_DIR := $(VORTEX_HOME)/hw/rtl
SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts
FOLDSIM_RTL_DIR := $(RTL_DIR)/afu/foldsim
VIVADO := $(XILINX_VIVADO)/bin/vivado
SLASHKIT ?= slashkit
# Parallel link runs; each Vivado of a large simulator needs tens of GB.
SLASHKIT_JOBS ?=

XLEN ?= 32
PREFIX ?= build$(XLEN)
TARGET ?= hw
DEVICE ?= xcv80-lsva4737-2MHP-e-S

ifneq ($(TARGET), hw)
$(error TARGET must be hw: the runtime's avedsim target simulates the generated design itself)
endif

BUILD_DIR := $(abspath $(PREFIX)_foldsim_$(TARGET))
# The runtime's in-process simulator generates the design in a directory of
# its own.
ifneq ($(FOLDSIM_WORK_DIR),)
BUILD_DIR := $(abspath $(FOLDSIM_WORK_DIR))
endif
BIN_DIR := $(BUILD_DIR)/bin
GEN_DIR := $(BUILD_DIR)/gen

# The classes threaded by FAME-5, and the units given plain-port boundaries
# first so they can be classes (lift_boundaries.py; <unit>[:drop=P,...][;port=P,...]).
FOLDSIM_CLASSES ?= VX_socket__lift
FOLDSIM_LIFT ?= VX_socket:drop=INSTANCE_ID
# Where circt-verilog, circt-opt and firtool are (hw/syn/common.mk's CIRCT);
# the foldsim driver is hw/syn/common.mk's FOLDSIM.
CIRCT_BIN ?= $(CIRCT_PATH)/bin
LIFT_PYTHON ?= python3
# The kernel clock (MHz): the image is timed at it and the shell programs it.
FOLDSIM_FREQ ?= 250
# Options of the foldsim driver: by default the set that times s2w4i2 at
# 250 MHz (regions of 2048 loads with their own control, registered state
# committed by select over two host cycles, pipelined multipliers, staged
# memory writes and tokens, folded fixed-latency pipelines); docs/designs/
# foldsim.md lists each. Large images add --compact-staging; multi-SLR ones
# --domains and --relay.
FOLDSIM_FLAGS ?= --region-loads 2048 --registered-state --commit-select --state-stages 2 \
                 --pipeline-multipliers --staged-ram-writes --staged-tokens --fold-pipelines

# The target: Vortex with its AXI4 port, DCR, start and busy as plain ports.
TARGET_TOP := foldsim_target

# The same tree the native synthesis flows elaborate.
CONFIGS ?=
RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs \
               -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm \
               -I$(RTL_DIR)/cache -I$(RTL_DIR)/fpu -I$(RTL_DIR)/afu/common -I$(FOLDSIM_RTL_DIR)/target
SYNTH_DEFINES := -DSYNTHESIS -DVIVADO -DNDEBUG
ifdef PERF
PLATFORM_DEFINES := -DPERF_ENABLE
endif
# Every default is the configuration's (VX_config.toml with CONFIGS). The FPU
# type is chosen as in the native Vivado flows: DSP selects the Xilinx
# floating-point cores, which the simulator keeps as instances: their netlists
# are encrypted, so their state cannot be threaded, and each is simulated by
# one unthreaded copy of itself and a per-thread store of its results (foldsim
# --fold-pipelines, by the pipeline contract xilinx_ip_stubs.sv declares).
XCONFIGS := $(shell python3 $(VORTEX_HOME)/ci/gen_config.py \
	--config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS) -DVX_CFG_XLEN=$(XLEN) $(SYNTH_DEFINES) $(PLATFORM_DEFINES)')
TYPE_DEFINES := $(filter -DVX_CFG_TCU_TYPE_% -DVX_CFG_FPU_TYPE_%,$(XCONFIGS))
cfg_value = $(lastword $(patsubst -D$(1)=%,%,$(filter -D$(1)=%,$(XCONFIGS))))
# One memory port per memory bank of the configuration, each its own memory
# endpoint and DRAM channel. AXI ids are what Vortex drives with its tag
# buffer engaged: the buffer's 4 index bits, and above them the index of the
# GPU's memory port the read came from, of which there are at most as many as
# memory ports.
MEM_PORTS := $(call cfg_value,VX_CFG_PLATFORM_MEMORY_NUM_BANKS)
ifeq ($(filter 1 2 4 8,$(MEM_PORTS)),)
$(error the foldsim kernel has 1, 2, 4 or 8 memory ports (VX_CFG_PLATFORM_MEMORY_NUM_BANKS=$(MEM_PORTS)))
endif
MEM_ID_W := $(shell python3 -c 'print(4 + ($(MEM_PORTS) - 1).bit_length())')
PLATFORM_DEFINES += -DPLATFORM_MEMORY_ID_WIDTH=$(MEM_ID_W)
# The shift that takes an address to its memory port's channel: the ports
# interleave by block, or (INTERLEAVE=0) split the space, where every port's
# channel holds the address as it is.
MEM_CH_SHIFT := $(if $(filter 1,$(call cfg_value,VX_CFG_PLATFORM_MEMORY_INTERLEAVE)), \
                $(shell python3 -c 'print(($(MEM_PORTS) - 1).bit_length())'),0)
# The extensions' packages and include paths, as every synthesis flow finds
# them.
include $(VORTEX_HOME)/hw/syn/extensions.mk
GEN_CFLAGS := -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN) $(CONFIGS) $(TYPE_DEFINES) $(PLATFORM_DEFINES) \
              $(SYNTH_DEFINES) $(RTL_INCLUDE)

# Rewritten only when a setting changes, so an unchanged build stays current.
define stamp
$(shell mkdir -p $(BUILD_DIR) && echo '$(2)' > $(1).tmp && \
        (cmp -s $(1).tmp $(1) && rm -f $(1).tmp || mv -f $(1).tmp $(1)))
endef
CONFIG_STAMP := $(BUILD_DIR)/cflags.stamp
CLASSES_STAMP := $(BUILD_DIR)/classes.stamp
$(call stamp,$(CONFIG_STAMP),$(GEN_CFLAGS))
$(call stamp,$(CLASSES_STAMP),$(FOLDSIM_CLASSES) | $(FOLDSIM_LIFT)$(if $(FOLDSIM_FLAGS), | $(FOLDSIM_FLAGS)))

SRCS_NAME := vortex_srcs
SRCS_DIR := $(BUILD_DIR)/$(SRCS_NAME)
SRCS_LIST := $(BUILD_DIR)/$(SRCS_NAME).f
LIFT_LIST := $(BUILD_DIR)/lift/srcs.f
TARGET_SRCS := $(if $(strip $(FOLDSIM_LIFT)),$(LIFT_LIST),$(SRCS_LIST))
GEN_SV := $(GEN_DIR)/foldsim_sim.sv
KERNEL_VH := $(GEN_DIR)/foldsim_kernel.vh
KERNEL_SRCS := $(BUILD_DIR)/kernel_srcs.f
IP_DIR := $(BUILD_DIR)/foldsim_afu_ip
CFG := $(BUILD_DIR)/config.cfg
VBIN := $(BIN_DIR)/foldsim.vbin

# The kernel's own sources are not the target's: they are read only when the
# kernel is packaged or emulated.
KERNEL_RTL := $(wildcard $(FOLDSIM_RTL_DIR)/*.sv $(FOLDSIM_RTL_DIR)/*.vh)
RTL_SRCS := $(filter-out $(KERNEL_RTL),$(shell find $(RTL_DIR) -name '*.sv' -o -name '*.v' -o -name '*.vh')) \
            $(wildcard $(ROOT_DIR)/hw/VX_config.vh) $(wildcard $(ROOT_DIR)/hw/VX_types.vh) \
            $(firstword $(MAKEFILE_LIST)) $(CONFIG_STAMP)
# The AFU shells of the native flows are second tops, and the foldsim kernel is
# compiled apart from the target.
SHELL_MODULES := $(notdir $(wildcard $(RTL_DIR)/afu/common/*.sv) $(wildcard $(FOLDSIM_RTL_DIR)/*.sv))

.PHONY: all sources generate package-ip ooc clean FORCE

all: $(VBIN)

# The Xilinx floating-point cores, declared as black boxes with their pipeline
# contracts; their contracts agree with the cores' generation and the RTL.
IP_STUBS := $(SCRIPT_DIR)/xilinx_ip_stubs.sv
# The list is written to a temporary and moved into place last, so a failing
# step leaves no list that looks current.
sources: $(SRCS_LIST)
$(SRCS_LIST): $(RTL_SRCS) $(IP_STUBS) $(SCRIPT_DIR)/xilinx_ip_gen.tcl
	rm -rf $(SRCS_DIR) $@ $@.tmp
	python3 $(SCRIPT_DIR)/check_ip_contracts.py --cflags '$(GEN_CFLAGS)'
	cd $(BUILD_DIR) && $(SCRIPT_DIR)/gen_sources.sh -P $(GEN_CFLAGS) -T$(TARGET_TOP) -C$(SRCS_NAME) \
		-O$(SRCS_NAME).f.tmp $(RTL_PKGS)
	cp $(IP_STUBS) $(SRCS_DIR)/ && echo "$(SRCS_DIR)/$(notdir $(IP_STUBS))" >> $@.tmp
	@for f in $(SHELL_MODULES); do \
		rm -f $(SRCS_DIR)/$$f; \
		grep -v "/$$f$$" $@.tmp > $@.tmp2 && mv $@.tmp2 $@.tmp; \
	done
	@if grep -qE 'localparam +VX_MEM_PORTS *= *0 *;' $(SRCS_DIR)/VX_gpu_pkg.sv; then \
		echo "ERROR: generated sources resolve VX_MEM_PORTS to 0 (no memory interface)"; exit 1; \
	fi
	mv $@.tmp $@

$(LIFT_LIST): $(SRCS_LIST) $(SRC_DIR)/lift_boundaries.py $(CLASSES_STAMP)
	rm -rf $(BUILD_DIR)/lift
	$(LIFT_PYTHON) $(SRC_DIR)/lift_boundaries.py --srcs $(SRCS_LIST) --top $(TARGET_TOP) \
		--out $(BUILD_DIR)/lift $(foreach u,$(FOLDSIM_LIFT),--unit '$(u)')

generate: $(GEN_SV) $(KERNEL_VH)
# The compiler is the driver and the tools it runs: foldsim-opt beside it and
# CIRCT's, so a rebuilt tool regenerates the simulator.
FOLDSIM_TOOLS := $(FOLDSIM) $(wildcard $(dir $(FOLDSIM))foldsim-opt \
                 $(addprefix $(CIRCT_BIN)/,circt-verilog circt-opt firtool))
$(GEN_SV): $(TARGET_SRCS) $(FOLDSIM_TOOLS) $(CLASSES_STAMP)
	$(FOLDSIM) --srcs $(TARGET_SRCS) --top $(TARGET_TOP) --out $(GEN_DIR) \
		$(if $(CIRCT_BIN),--circt-bin $(CIRCT_BIN)) $(FOLDSIM_FLAGS) $(FOLDSIM_CLASSES)

# What the kernel's top needs of the configuration: its memory ports, their id
# width, and FOLDSIM_MEM_REPEAT(f,s), `f(p) for each port p joined by `s, which
# lists the ports whose names carry their index. Spelled out here: Vivado's IP
# packager does not paste a macro's expanded argument into a macro name.
MEM_REPEAT := $(shell python3 -c 'print(" `s ".join("`f(%d)" % p for p in range($(MEM_PORTS))))')
$(KERNEL_VH): FORCE
	@mkdir -p $(@D)
	@printf '%s\n' '// Generated by hw/syn/foldsim. Do not edit.' \
		'`define FOLDSIM_MEM_PORTS $(MEM_PORTS)' '`define FOLDSIM_MEM_CH_SHIFT $(strip $(MEM_CH_SHIFT))' \
		'`define FOLDSIM_MEM_ID_W $(MEM_ID_W)' '`define FOLDSIM_MEM_REPEAT(f,s) $(MEM_REPEAT)' \
		$(if $(findstring --axi-spec,$(FOLDSIM_FLAGS)),'`define FOLDSIM_MEM_SPEC $(if $(findstring --axi-spec=2,$(FOLDSIM_FLAGS)),2,1)') > $@.tmp
	@cmp -s $@.tmp $@ && rm -f $@.tmp || mv -f $@.tmp $@

$(KERNEL_SRCS): $(GEN_SV) $(KERNEL_VH)
	@{ echo "+incdir+$(FOLDSIM_RTL_DIR)"; echo "+incdir+$(GEN_DIR)"; \
	   echo "$(GEN_SV)"; \
	   for f in foldsim_token.sv foldsim_mem_bmap.sv foldsim_mem_bank.sv foldsim_mem.sv foldsim_ctrl.sv \
	            foldsim_afu.sv; do echo "$(FOLDSIM_RTL_DIR)/$$f"; done; \
	 } > $@

# The kernel's clock at FOLDSIM_FREQ, for its out-of-context synthesis.
OOC_XDC := $(BUILD_DIR)/foldsim_afu_ooc.xdc
$(OOC_XDC): FORCE
	@mkdir -p $(@D)
	@printf 'create_clock -name ap_clk -period %s [get_ports ap_clk]\n' \
		$$(python3 -c 'print(round(1000 / $(FOLDSIM_FREQ), 3))') > $@.tmp
	@cmp -s $@.tmp $@ && rm -f $@.tmp || mv -f $@.tmp $@

package-ip: $(IP_DIR)/component.xml
# The kernel is packaged with the floating-point cores the simulator
# instantiates, generated as the aved flow generates them.
$(IP_DIR)/component.xml: $(KERNEL_SRCS) $(KERNEL_VH) $(OOC_XDC) $(wildcard $(FOLDSIM_RTL_DIR)/*) $(SCRIPT_DIR)/xilinx_ip_gen.tcl
	cd $(BUILD_DIR) && TOOL_DIR=$(SCRIPT_DIR) $(VIVADO) -mode batch -nojournal -nolog \
		-source $(VORTEX_HOME)/hw/syn/xilinx/aved/package_ip.tcl \
		-tclargs foldsim_afu $(KERNEL_SRCS) . $(DEVICE) ooc-xdc=$(OOC_XDC) \
		$$(grep -qE '^ *xil_f(ma|div|sqrt|mul|add) ' $(GEN_SV) || echo no-vendor-ip)

# foldsim --slr-groups writes a floorplan that keeps each threaded copy whole
# in one SLR; the implementation applies it before opt_design (floorplan.tcl).
FLOORPLAN_ENV := $(if $(findstring --slr-groups,$(FOLDSIM_FLAGS)),FOLDSIM_FLOORPLAN=$(GEN_DIR)/foldsim_floorplan.tcl)

# The kernel alone, out of context at FOLDSIM_FREQ: synthesis with global
# retiming, placement and routing, without the shell, its reports in
# $(BUILD_DIR)/ooc. It answers what a change does to the kernel's Fmax in
# about half an image build's time.
OOC_DIR := $(BUILD_DIR)/ooc
ooc: $(KERNEL_SRCS)
	rm -rf $(OOC_DIR) && mkdir -p $(OOC_DIR)
	cd $(OOC_DIR) && TOOL_DIR=$(SCRIPT_DIR) $(FLOORPLAN_ENV) $(VIVADO) -mode batch -nojournal -log vivado.log \
		-source $(SRC_DIR)/ooc.tcl -tclargs $(KERNEL_SRCS) $(DEVICE) \
		$$(python3 -c 'print(round(1000 / $(FOLDSIM_FREQ), 3))') \
		$$(grep -qE '^ *xil_f(ma|div|sqrt|mul|add) ' $(GEN_SV) || echo no-vendor-ip)
	@grep -A2 'WNS(ns)' $(OOC_DIR)/routed_timing_summary.rpt | tail -1

$(CFG): FORCE
	@python3 $(SCRIPT_DIR)/gen_platform.py --flow=aved --device=$(DEVICE) --emit=cfg --kernel=foldsim_afu \
		--no-host --mem-ports=$(MEM_PORTS) --kernel-mhz=$(FOLDSIM_FREQ) -o $@.tmp
	@cmp -s $@.tmp $@ && rm -f $@.tmp || mv -f $@.tmp $@

FORCE:

$(VBIN): $(IP_DIR)/component.xml $(CFG)
	mkdir -p $(BIN_DIR)
	TOOL_DIR=$(SCRIPT_DIR) $(FLOORPLAN_ENV) $(SLASHKIT) link -c $(CFG) -p hw -o $@ -k $(IP_DIR)/component.xml \
		--vivado $(VIVADO) $(if $(SLASHKIT_JOBS),--jobs $(SLASHKIT_JOBS)) \
		--pre-synth-tcls $(SRC_DIR)/pre_synth_hook.tcl
	@echo "image: $@"

clean:
	rm -rf $(BUILD_DIR)
