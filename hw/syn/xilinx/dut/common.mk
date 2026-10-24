ROOT_DIR := $(realpath ../../../../..)
include $(ROOT_DIR)/config.mk
# Exports CIRCT_VERILOG as an absolute path. gen_sources.sh resolves its
# preprocessor from the environment, so without this the flow silently depends
# on circt-verilog being on PATH.
include $(VORTEX_HOME)/hw/syn/common.mk

DEVICE ?= xcu55c-fsvh2892-2L-e

MAX_JOBS ?= 8

# Synthesis optimization level (standardized across hw/syn):
#   0    -- fastest compile, minimal optimization (Flow_RuntimeOptimized)
#   1, 2 -- Vivado defaults
#   3    -- default: aggressive performance strategy
# project.tcl reads this through the OPT_LEVEL env var.
OPT_LEVEL ?= 3

VIVADO := $(XILINX_VIVADO)/bin/vivado

SRC_DIR := $(VORTEX_HOME)/hw/syn/xilinx/dut

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi
AFU_DIR := $(RTL_DIR)/afu/xrt
AFU_COMMON_DIR := $(RTL_DIR)/afu/common
SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts
UNITTEST_DIR := $(VORTEX_HOME)/hw/unittest

NCPUS := $(shell lscpu | grep "^Core(s) per socket:" | awk '{print $$4}')
JOBS ?= $(shell echo $$(( $(NCPUS) > $(MAX_JOBS) ? $(MAX_JOBS) : $(NCPUS) )))

# `override`: these are facts about this flow, not preferences. A caller that
# sets CONFIGS on the MAKE COMMAND LINE would otherwise kill every `CONFIGS +=`
# here and in build.mk silently -- command-line variables win over `+=`, and
# nothing warns. With `override`, a command-line CONFIGS is extra defines on
# top, which is what it reads as.
override CONFIGS += -DSYNTHESIS -DVIVADO -DNDEBUG

XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py --config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS) -DVX_CFG_XLEN=$(XLEN)')

CFLAGS += -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN)
# raw CONFIGS carries only an enum *value* (e.g. VX_CFG_TCU_TYPE=TFR); append the
# resolved type selectors so an enum override selects its RTL implementation.
CFLAGS += $(CONFIGS) $(filter -DVX_CFG_TCU_TYPE_% -DVX_CFG_FPU_TYPE_%,$(XCONFIGS))
CFLAGS += $(RTL_INCLUDE)

# Power analysis via SAIF switching-activity annotation.
# SAIF_FILE : path to the SAIF file produced by rtlsim with SAIF=1 (required for 'power' target)
# SAIF_INST : instance path of the DUT inside the simulation hierarchy, used to
#             strip the testbench prefix from SAIF signal names so they align with
#             the synthesized netlist (e.g. "TOP.rtlsim_shim.vortex").
#             Leave empty when the SAIF root scope already matches the top module.
SAIF_FILE ?=
SAIF_INST ?=

# Build targets
all: $(PROJECT).xpr

gen-sources: project_1/sources.txt
project_1/sources.txt:
	mkdir -p project_1
	$(SCRIPT_DIR)/gen_sources.sh $(CFLAGS) -T$(TOP_LEVEL_ENTITY) -P -Cproject_1/src -Oproject_1/sources.txt $(RTL_PKGS) \
		-E -X$(SCRIPT_DIR)/xilinx_ip_stubs.sv

build: $(PROJECT).xpr
$(PROJECT).xpr: project_1/sources.txt
ifdef FPU_IP
	MAX_JOBS=$(JOBS) OPT_LEVEL=$(OPT_LEVEL) FPU_IP=project_1/ip TOOL_DIR=$(SCRIPT_DIR) $(VIVADO) -mode batch -source $(SRC_DIR)/project.tcl -tclargs $(TOP_LEVEL_ENTITY) $(DEVICE) project_1/sources.txt $(SRC_DIR)/project.xdc
else
	MAX_JOBS=$(JOBS) OPT_LEVEL=$(OPT_LEVEL) TOOL_DIR=$(SCRIPT_DIR) $(VIVADO) -mode batch -source $(SRC_DIR)/project.tcl -tclargs $(TOP_LEVEL_ENTITY) $(DEVICE) project_1/sources.txt $(SRC_DIR)/project.xdc
endif

# Re-run power analysis on an existing post-implementation checkpoint.
# Requires SAIF_FILE=<path>.  Does not rebuild the design.
# Example: make power SAIF_FILE=/path/to/trace.saif SAIF_INST=TOP.rtlsim_shim.vortex
power:
	@if [ ! -f post_impl.dcp ]; then \
	  echo "ERROR: post_impl.dcp not found. Run 'make build' first."; exit 1; \
	fi
	@if [ -z "$(SAIF_FILE)" ]; then \
	  echo "ERROR: SAIF_FILE not specified. Usage: make power SAIF_FILE=<path/to/trace.saif>"; exit 1; \
	fi
	TOOL_DIR=$(SCRIPT_DIR) SAIF_FILE=$(SAIF_FILE) SAIF_INST=$(SAIF_INST) \
	  $(VIVADO) -mode batch -source $(SCRIPT_DIR)/xilinx_power_analysis.tcl

# The logic screen (foldsim-estimate-timing): the logic levels of every
# register-to-register path of the design, counted as Vivado counts them on
# DEVICE's family (hw/syn/xilinx/platforms), by RTL unit, in seconds and
# without Vivado. It elaborates the same sources and
# defines as gen-sources into a CIRCT netlist of its own, with the Xilinx
# floating-point cores as black boxes. ci/logic_gate.py gates
# screen_summary.csv and screen_units.json; screen_paths.txt has every path
# of at least SCREEN_PATH_LEVELS (screen_summary.py's watch threshold), op by
# op, by unit.
SCREEN_PATH_LEVELS ?= 8
screen:
	mkdir -p screen
	$(SCRIPT_DIR)/gen_sources.sh $(CFLAGS) -T$(TOP_LEVEL_ENTITY) -Oscreen/sources.f $(RTL_PKGS)
	$(SCRIPT_DIR)/circt_elaborate.sh -f mlir -t $(TOP_LEVEL_ENTITY) -l screen/sources.f \
		-x $(SCRIPT_DIR)/xilinx_ip_stubs.sv -r screen/memories.rpt -o screen/netlist.mlir
	$(FOLDSIM_OPT) screen/netlist.mlir --pass-pipeline='builtin.module(foldsim-screen-map)' \
		--emit-bytecode -o screen/mapped.mlirbc
	family=$$(python3 $(SCRIPT_DIR)/gen_platform.py --device=$(DEVICE) --emit=family) && \
	$(FOLDSIM_OPT) screen/mapped.mlirbc -o /dev/null --pass-pipeline="builtin.module(foldsim-estimate-timing{top=$(TOP_LEVEL_ENTITY) family=$$family output=screen/endpoints.csv instances=screen/instances.csv report=screen/report.txt paths=screen/paths.jsonl path-levels=$(SCREEN_PATH_LEVELS)})"
	python3 $(SCRIPT_DIR)/screen_summary.py screen --top $(TOP_LEVEL_ENTITY)

clean:
ifndef RESUME
	rm -rf project_1 screen screen_summary.csv screen_units.json screen_paths.txt
	rm -rf .Xil
	rm -f *.rpt
	rm -f *.log
	rm -f *.jou
	rm -f *.dcp
else
	@echo "RESUME is defined, skipping clean."
endif

.PHONY: all gen-sources build screen clean power
