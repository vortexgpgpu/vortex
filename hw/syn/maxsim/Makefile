# maxsim image build for the V80, driven as the native aved flow
# (hw/syn/xilinx/aved):
#
#   make [DEVICE=<part>] [PREFIX=...] [CONFIGS="-DVX_CFG_..."] [PERF=1]
#        [MAXSIM_CLASSES="..."] [MAXSIM_LIFT="..."] [CIRCT_BIN=<dir>]
#
# CIRCT compiles Vortex into a simulator (maxsim.py): each class threaded
# through one physical copy (FAME-5), the rest of the target a model of its
# own, and every channel between them in one token fabric. The kernel
# (hw/rtl/afu/maxsim) adds the memory and control endpoints around it. The
# image lands in $(BUILD_DIR)/bin/maxsim.vbin; run it with
#   FPGA_BIN_DIR=<that bin> ./ci/blackbox.sh --driver=maxsim --target=hw ...
# The runtime's avedsim target simulates the same generated design in-process.

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk
include $(VORTEX_HOME)/hw/syn/common.mk

SRC_DIR := $(VORTEX_HOME)/hw/syn/maxsim
RTL_DIR := $(VORTEX_HOME)/hw/rtl
SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts
MAXSIM_RTL_DIR := $(RTL_DIR)/afu/maxsim
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

BUILD_DIR := $(abspath $(PREFIX)_maxsim_$(TARGET))
# The runtime's in-process simulator generates the design in a directory of
# its own.
ifneq ($(MAXSIM_WORK_DIR),)
BUILD_DIR := $(abspath $(MAXSIM_WORK_DIR))
endif
BIN_DIR := $(BUILD_DIR)/bin
GEN_DIR := $(BUILD_DIR)/gen

# The classes threaded by FAME-5, and the units given plain-port boundaries
# first so they can be classes (lift_boundaries.py; <unit>[:drop=P,...][;port=P,...]).
MAXSIM_CLASSES ?= VX_socket__lift
MAXSIM_LIFT ?= VX_socket:drop=INSTANCE_ID;port=SOCKET_ID
# Where circt-verilog, circt-opt and firtool are (hw/syn/common.mk's CIRCT).
CIRCT_BIN ?= $(CIRCT_PATH)/bin
LIFT_PYTHON ?= python3
# The kernel clock (MHz): the image is timed at it and the shell programs it.
MAXSIM_FREQ ?= 143

# The target: Vortex with its AXI4 port, DCR, start and busy as plain ports.
TARGET_TOP := VX_firesim_wrap

# The same tree the native synthesis flows elaborate.
CONFIGS ?=
RTL_INCLUDE := -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs \
               -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm \
               -I$(RTL_DIR)/cache -I$(RTL_DIR)/fpu -I$(RTL_DIR)/afu/common -I$(RTL_DIR)/afu/firesim
SYNTH_DEFINES := -DSYNTHESIS -DVIVADO -DNDEBUG
# One memory port, as the memory endpoint models; 4 AXI id bits, what Vortex
# drives with its tag buffer engaged. The FPU and RTU use Vortex's own
# datapaths (the STD FPU, the RTU's soft cores with their DSP multiplies): a
# vendor netlist is not source the flow can read, nor state it can thread. The
# staged sources therefore never select the floating-point cores (see
# VENDOR_FP_UNITS), and the RTU divider has the soft one's latency.
PLATFORM_DEFINES := -DVX_CFG_PLATFORM_MEMORY_NUM_BANKS=1 -DPLATFORM_MEMORY_ID_WIDTH=4 \
                    -DVX_CFG_RTU_FDIV_LATENCY=17 -DVX_CFG_FPU_TYPE_STD
ifdef PERF
PLATFORM_DEFINES += -DPERF_ENABLE
endif
XCONFIGS := $(shell python3 $(VORTEX_HOME)/ci/gen_config.py \
	--config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS) -DVX_CFG_XLEN=$(XLEN) $(SYNTH_DEFINES) $(PLATFORM_DEFINES)')
TYPE_DEFINES := $(filter -DVX_CFG_TCU_TYPE_% -DVX_CFG_FPU_TYPE_%,$(XCONFIGS))
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
$(call stamp,$(CLASSES_STAMP),$(MAXSIM_CLASSES) | $(MAXSIM_LIFT))

SRCS_NAME := vortex_srcs
SRCS_DIR := $(BUILD_DIR)/$(SRCS_NAME)
SRCS_LIST := $(BUILD_DIR)/$(SRCS_NAME).f
LIFT_LIST := $(BUILD_DIR)/lift/srcs.f
TARGET_SRCS := $(if $(strip $(MAXSIM_LIFT)),$(LIFT_LIST),$(SRCS_LIST))
GEN_SV := $(GEN_DIR)/maxsim_sim.sv
KERNEL_SRCS := $(BUILD_DIR)/kernel_srcs.f
IP_DIR := $(BUILD_DIR)/maxsim_afu_ip
CFG := $(BUILD_DIR)/config.cfg
VBIN := $(BIN_DIR)/maxsim.vbin

RTL_SRCS := $(shell find $(RTL_DIR) -name '*.sv' -o -name '*.v' -o -name '*.vh') \
            $(wildcard $(ROOT_DIR)/hw/VX_config.vh) $(wildcard $(ROOT_DIR)/hw/VX_types.vh) \
            $(firstword $(MAKEFILE_LIST)) $(CONFIG_STAMP)
# The AFU shells of the native flows are second tops, and the maxsim kernel is
# compiled apart from the target.
SHELL_MODULES := $(notdir $(wildcard $(RTL_DIR)/afu/common/*.sv) $(wildcard $(MAXSIM_RTL_DIR)/*.sv))

.PHONY: all sources generate package-ip clean FORCE

all: $(VBIN)

# The flow threads memories itself, so an asynchronous-read RAM stays one
# (VX_async_ram_patch.sv) instead of becoming placeholders for a Vivado hook.
# The units that would wrap a Xilinx floating-point core take their soft path,
# the one they take wherever no vendor core exists.
VENDOR_FP_UNITS := VX_fma_unit.sv VX_fdiv_unit.sv VX_fsqrt_unit.sv
sources: $(SRCS_LIST)
$(SRCS_LIST): $(RTL_SRCS) $(SRC_DIR)/VX_async_ram_patch.sv
	rm -rf $(SRCS_DIR)
	cd $(BUILD_DIR) && $(SCRIPT_DIR)/gen_sources.sh -P $(GEN_CFLAGS) -T$(TARGET_TOP) -C$(SRCS_NAME) \
		-O$(SRCS_NAME).f $(RTL_PKGS)
	cp $(SRC_DIR)/VX_async_ram_patch.sv $(SRCS_DIR)/VX_async_ram_patch.sv
	@for f in $(VENDOR_FP_UNITS); do \
		[ -f $(SRCS_DIR)/$$f ] || continue; \
		grep -q 'localparam VENDOR_OK = 1;' $(SRCS_DIR)/$$f || \
			{ echo "ERROR: $$f no longer selects its vendor core by VENDOR_OK"; exit 1; }; \
		sed -i 's/localparam VENDOR_OK = 1;/localparam VENDOR_OK = 0;/' $(SRCS_DIR)/$$f; \
	done
	@for f in $(SHELL_MODULES); do \
		rm -f $(SRCS_DIR)/$$f; \
		grep -v "/$$f$$" $@ > $@.tmp && mv $@.tmp $@; \
	done
	@if grep -qE 'localparam +VX_MEM_PORTS *= *0 *;' $(SRCS_DIR)/VX_gpu_pkg.sv; then \
		echo "ERROR: generated sources resolve VX_MEM_PORTS to 0 (no memory interface)"; exit 1; \
	fi

$(LIFT_LIST): $(SRCS_LIST) $(SRC_DIR)/lift_boundaries.py $(CLASSES_STAMP)
	rm -rf $(BUILD_DIR)/lift
	$(LIFT_PYTHON) $(SRC_DIR)/lift_boundaries.py --srcs $(SRCS_LIST) --top $(TARGET_TOP) \
		--out $(BUILD_DIR)/lift $(foreach u,$(MAXSIM_LIFT),--unit '$(u)')

generate: $(GEN_SV)
$(GEN_SV): $(TARGET_SRCS) $(SRC_DIR)/maxsim.py $(CLASSES_STAMP)
	python3 $(SRC_DIR)/maxsim.py --srcs $(TARGET_SRCS) --top $(TARGET_TOP) --out $(GEN_DIR) \
		$(if $(CIRCT_BIN),--circt-bin $(CIRCT_BIN)) $(MAXSIM_CLASSES)

$(KERNEL_SRCS): $(GEN_SV)
	@{ echo "+incdir+$(MAXSIM_RTL_DIR)"; echo "+incdir+$(GEN_DIR)"; \
	   echo "$(GEN_SV)"; \
	   for f in maxsim_token.sv maxsim_mem.sv maxsim_ctrl.sv maxsim_afu.sv; do echo "$(MAXSIM_RTL_DIR)/$$f"; done; \
	 } > $@

package-ip: $(IP_DIR)/component.xml
$(IP_DIR)/component.xml: $(KERNEL_SRCS) $(wildcard $(MAXSIM_RTL_DIR)/*)
	cd $(BUILD_DIR) && TOOL_DIR=$(SCRIPT_DIR) $(VIVADO) -mode batch -nojournal -nolog \
		-source $(VORTEX_HOME)/hw/syn/xilinx/aved/package_ip.tcl \
		-tclargs maxsim_afu $(KERNEL_SRCS) . $(DEVICE) no-vendor-ip

$(CFG): FORCE
	@python3 $(SCRIPT_DIR)/gen_platform.py --flow=aved --device=$(DEVICE) --emit=cfg --kernel=maxsim_afu \
		--no-host --kernel-mhz=$(MAXSIM_FREQ) -o $@.tmp
	@cmp -s $@.tmp $@ && rm -f $@.tmp || mv -f $@.tmp $@

FORCE:

$(VBIN): $(IP_DIR)/component.xml $(CFG)
	mkdir -p $(BIN_DIR)
	TOOL_DIR=$(SCRIPT_DIR) $(SLASHKIT) link -c $(CFG) -p hw -o $@ -k $(IP_DIR)/component.xml \
		--vivado $(VIVADO) $(if $(SLASHKIT_JOBS),--jobs $(SLASHKIT_JOBS))
	@echo "image: $@"

clean:
	rm -rf $(BUILD_DIR)
