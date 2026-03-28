ROOT_DIR := $(realpath ../../../../../..)
include $(ROOT_DIR)/config.mk

DEVICE ?= xcu55c-fsvh2892-2L-e

MAX_JOBS ?= 8

VIVADO := $(XILINX_VIVADO)/bin/vivado

SRC_DIR := $(VORTEX_HOME)/hw/syn/xilinx/dut

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi
AFU_DIR := $(RTL_DIR)/afu/xrt
SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts
UNITTEST_DIR := $(VORTEX_HOME)/hw/unittest

NCPUS := $(shell lscpu | grep "^Core(s) per socket:" | awk '{print $$4}')
JOBS ?= $(shell echo $$(( $(NCPUS) > $(MAX_JOBS) ? $(MAX_JOBS) : $(NCPUS) )))

CONFIGS += -DSYNTHESIS -DVIVADO -DNDEBUG

XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py --config $(VORTEX_HOME)/hw/VX_config.toml --cflags '$(CONFIGS)')

# Power analysis via VCD switching-activity annotation.
# VCD      : path to the VCD file produced by rtlsim (required for 'power' target)
# VCD_INST : instance path of the DUT inside the simulation hierarchy, used to
#            strip the testbench prefix from VCD signal names so they align with
#            the synthesized netlist (e.g. "TOP.Vortex").
#            Leave empty when the VCD root scope already matches the top module.
VCD      ?=
VCD_INST ?=

# Build targets
all: $(PROJECT).xpr

gen-sources: project_1/sources.txt
project_1/sources.txt:
	mkdir -p project_1
	$(SCRIPT_DIR)/gen_sources.sh $(CONFIGS) $(RTL_INCLUDE) -T$(TOP_LEVEL_ENTITY) -P -Cproject_1/src -Oproject_1/sources.txt

build: $(PROJECT).xpr
$(PROJECT).xpr: project_1/sources.txt
ifdef FPU_IP
	MAX_JOBS=$(JOBS) FPU_IP=project_1/ip TOOL_DIR=$(SCRIPT_DIR) VCD_FILE=$(VCD) VCD_INST=$(VCD_INST) $(VIVADO) -mode batch -source $(SRC_DIR)/project.tcl -tclargs $(TOP_LEVEL_ENTITY) $(DEVICE) project_1/sources.txt $(SRC_DIR)/project.xdc
else
	MAX_JOBS=$(JOBS) TOOL_DIR=$(SCRIPT_DIR) VCD_FILE=$(VCD) VCD_INST=$(VCD_INST) $(VIVADO) -mode batch -source $(SRC_DIR)/project.tcl -tclargs $(TOP_LEVEL_ENTITY) $(DEVICE) project_1/sources.txt $(SRC_DIR)/project.xdc
endif

# Re-run power analysis on an existing post-implementation checkpoint.
# Requires VCD=<path>.  Does not rebuild the design.
# Example: make power VCD=/path/to/sim.vcd VCD_INST=TOP.Vortex
power:
	@if [ ! -f project_1/post_impl.dcp ]; then \
	  echo "ERROR: project_1/post_impl.dcp not found. Run 'make build' first."; exit 1; \
	fi
	@if [ -z "$(VCD)" ]; then \
	  echo "ERROR: VCD not specified. Usage: make power VCD=<path/to/sim.vcd>"; exit 1; \
	fi
	TOOL_DIR=$(SCRIPT_DIR) VCD_FILE=$(VCD) VCD_INST=$(VCD_INST) \
	  $(VIVADO) -mode batch -source $(SRC_DIR)/power_analysis.tcl

clean:
ifndef RESUME
	rm -rf project_1
	rm -rf .Xil
	rm -f *.rpt
	rm -f *.log
	rm -f *.jou
	rm -f *.dcp
else
	@echo "RESUME is defined, skipping clean."
endif

.PHONY: all gen-sources build clean power