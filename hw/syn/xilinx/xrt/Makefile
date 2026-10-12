ROOT_DIR := $(realpath ../../../..)
include $(ROOT_DIR)/config.mk
# Exports CIRCT_VERILOG as an absolute path. gen_sources.sh resolves its
# preprocessor from the environment, so without this the flow silently depends
# on circt-verilog being on PATH.
include $(VORTEX_HOME)/hw/syn/common.mk

ifneq ($(findstring Makefile, $(MAKEFILE_LIST)), Makefile)
help:
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<hw/hw_emu> PLATFORM=<FPGA platform>"
	$(ECHO) "      Command to generate the design for specified Target and Device."
	$(ECHO) ""
	$(ECHO) "  make clean"
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
endif

TARGET ?= hw
PLATFORM ?=

PREFIX ?= build$(XLEN)

SAIF_FILE ?=
SAIF_INST ?=

# Synthesis optimization level (standardized across hw/syn):
#   0 -- fastest compile, no optimization (Vitis --optimize 0)
#   3 -- default
# Flows through to v++ via --optimize. DEBUG builds force level 0.
OPT_LEVEL ?= 3

SRC_DIR := $(VORTEX_HOME)/hw/syn/xilinx/xrt
AFU_DIR := $(VORTEX_HOME)/hw/rtl/afu/xrt

VPP := $(XILINX_VITIS)/bin/v++

PLATFORM_TO_XSA = $(strip $(patsubst %.xpfm, % , $(shell basename $(PLATFORM))))
XSA := $(call PLATFORM_TO_XSA, $(PLATFORM))

DEV_ARCH := $(shell platforminfo -p $(PLATFORM) | grep 'FPGA Family' | sed 's/.*://' | sed '/ai_engine/d' | sed 's/^[[:space:]]*//')

BUILD_DIR := $(PREFIX)_$(XSA)_$(TARGET)
BIN_DIR := $(BUILD_DIR)/bin

XO_CONTAINER := $(BIN_DIR)/vortex_afu.xo
XCLBIN_CONTAINER := $(BIN_DIR)/vortex_afu.xclbin

ifeq ($(TARGET), hw_emu)
	override CONFIGS += -DSIMULATION
endif

.DEFAULT_GOAL := all
include $(VORTEX_HOME)/hw/syn/xilinx/common.mk

# Kernel clock target frequency (MHz). The card descriptor supplies the
# default; override on the command line (e.g. KERNEL_FREQ=250) for extra
# timing-closure margin.

# Timing-miss policy. Default (1): a design that misses KERNEL_FREQ is
# auto-scaled down to the achieved frequency and the build succeeds with
# truthful metadata (the stock Alveo behavior). FREQ_SCALING=0 makes a
# scaled result unacceptable: any miss fails the build — for sign-off
# builds where the target is the contract. Either way the post-build gate
# verifies the implementation was actually timed at KERNEL_FREQ and that a
# scaled bitstream reports its real clock.
FREQ_SCALING ?= 1

# Kernel compiler global settings
VPP_FLAGS += --link --target $(TARGET) --platform $(PLATFORM) --save-temps --no_ip_cache
VPP_FLAGS += --vivado.synth.jobs $(JOBS) --vivado.impl.jobs $(JOBS)
VPP_FLAGS += --kernel_frequency $(KERNEL_FREQ)

# register compilation hooks
VPP_FLAGS += --xp "vivado_prop:run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=${SRC_DIR}/pre_opt_hook.tcl"

# Implementation strategy. Unset by default: vpl picks the platform's own.
# Name a Vivado impl strategy (e.g. Performance_ExplorePostRoutePhysOpt) to
# trade build time for timing closure on a design that is routing-bound.
ifdef IMPL_STRATEGY
VPP_FLAGS += --xp "vivado_prop:run.impl_1.STRATEGY=$(IMPL_STRATEGY)"
endif

# Card settings, resolved from the platform's XSA and FPGA family.
ifneq ($(PLATFORM),)
PLATFORM_GEN_ARGS := --flow=xrt --xsa=$(XSA) --family='$(DEV_ARCH)'
include $(VORTEX_HOME)/hw/syn/xilinx/platform.mk
endif

VPP_FLAGS += --report_level 2
VPP_FLAGS += --config $(SRC_DIR)/vitis.ini

# Generates profile summary report
ifdef PROFILE
	VPP_FLAGS += --profile_kernel data:all:all:all
	VPP_FLAGS += --profile_kernel stall:all:all:all
endif

# Debugging
ifdef DEBUG
	VPP_FLAGS += -g --optimize 0 --debug.protocol all
	ifeq ($(TARGET), hw)
		VPP_FLAGS += --debug.chipscope vortex_afu_1
		CFLAGS += -DNDEBUG -DCHIPSCOPE $(DBG_SCOPE_FLAGS)
	else
		VPP_FLAGS += --vivado.prop fileset.sim_1.xsim.elaborate.debug_level=all
		CFLAGS += -DVX_DBG_DEBUG_LEVEL=$(DEBUG) $(DBG_TRACE_FLAGS)
	endif
else
	VPP_FLAGS += --optimize $(OPT_LEVEL)
	CFLAGS += -DNDEBUG
endif

CFLAGS += $(COMPILE_CFLAGS)

# RTL Kernel only supports Hardware and Hardware Emulation.
ifneq ($(TARGET),$(findstring $(TARGET), hw hw_emu))
	$(warning WARNING:Application supports only hw hw_emu TARGET. Please use the target for running the application)
endif

.PHONY: all clean emconfig check-devices power

all: check-devices check-axil-blocks emconfig $(XCLBIN_CONTAINER) report

gen-xo: $(XO_CONTAINER)
$(XO_CONTAINER): $(BUILD_DIR)/sources.txt
	mkdir -p $(BUILD_DIR); cd $(BUILD_DIR);	TOOL_DIR=$(SCRIPT_DIR) $(VIVADO) -mode batch -source $(SRC_DIR)/gen_xo.tcl -tclargs ../$(XO_CONTAINER) vortex_afu sources.txt ../$(BUILD_DIR)

gen-bin: $(XCLBIN_CONTAINER)
$(XCLBIN_CONTAINER): $(XO_CONTAINER) $(SCOPE_JSON)
	mkdir -p $(BIN_DIR); cd $(BUILD_DIR); TOOL_DIR=$(SCRIPT_DIR) KERNEL_FREQ=$(KERNEL_FREQ) $(if $(IMPL_MAX_THREADS),IMPL_MAX_THREADS=$(IMPL_MAX_THREADS)) $(if $(VORTEX_SLR),VORTEX_SLR=$(VORTEX_SLR)) $(VPP) $(VPP_FLAGS) -o ../$(XCLBIN_CONTAINER) ../$(XO_CONTAINER)

emconfig: $(BIN_DIR)/emconfig.json
$(BIN_DIR)/emconfig.json:
	mkdir -p $(BIN_DIR); cd $(BUILD_DIR); emconfigutil --platform $(PLATFORM) --od ../$(BIN_DIR)

report: $(XCLBIN_CONTAINER)
ifeq ($(TARGET), hw)
	cp $(BUILD_DIR)/_x/logs/link/vivado.log $(BUILD_DIR)/bin
	cp $(BUILD_DIR)/_x/logs/link/syn/ulp_vortex_afu_1_0_synth_1_runme.log $(BUILD_DIR)/bin
	cp $(BUILD_DIR)/_x/reports/link/syn/ulp_vortex_afu_1_0_synth_1_ulp_vortex_afu_1_0_utilization_synth.rpt $(BUILD_DIR)/bin
	cp $(BUILD_DIR)/_x/reports/link/imp/impl_1_hw_bb_locked_utilization_placed.rpt $(BUILD_DIR)/bin
	cp $(BUILD_DIR)/_x/reports/link/imp/impl_1_hw_bb_locked_timing_summary_routed.rpt $(BUILD_DIR)/bin
	cp $(BUILD_DIR)/_x/link/vivado/vpl/prj/prj.runs/impl_1/hier_utilization.rpt $(BUILD_DIR)/bin
	python3 $(SCRIPT_DIR)/xilinx_check_kernel_clock.py --freq $(KERNEL_FREQ) \
		--report $(BUILD_DIR)/bin/impl_1_hw_bb_locked_timing_summary_routed.rpt \
		--new-clk-freq $(BUILD_DIR)/_x/link/int/_new_clk_freq \
		$(if $(filter 0,$(FREQ_SCALING)),--no-scaling)
endif

chipscope:
	debug_hw --vivado --host localhost --ltx_file $(BUILD_DIR)/bin/vortex_afu.ltx &

# Re-run power analysis on an existing build without rebuilding the design.
# The Vitis flow writes the routed checkpoint under _x/link/vivado/vpl/prj/prj.runs/impl_1/.
# Usage: make power SAIF_FILE=<saif-path> [SAIF_INST=<instance-path>] [BUILD_DIR=<path>]
power:
	@if [ -z "$(SAIF_FILE)" ]; then \
	  echo "ERROR: SAIF_FILE not specified. Usage: make power SAIF_FILE=<path/to/trace.saif>"; exit 1; \
	fi
	TOOL_DIR=$(SCRIPT_DIR) SAIF_FILE=$(SAIF_FILE) SAIF_INST=$(SAIF_INST) \
	  BUILD_DIR=$(BUILD_DIR) $(VIVADO) -mode batch -source $(SCRIPT_DIR)/xilinx_power_analysis.tcl

clean:
	$(RMDIR) $(BUILD_DIR)

# Check the devices avaiable
check-devices:
ifndef PLATFORM
	$(error PLATFORM not set. Please set the PLATFORM properly and rerun. Run "make help" for more details.)
endif
ifndef XILINX_VITIS
	$(error XILINX_VITIS variable is not set, please set correctly and rerun)
endif
