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

XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py --config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS)')

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
	$(SCRIPT_DIR)/gen_sources.sh $(CONFIGS) $(RTL_INCLUDE) -T$(TOP_LEVEL_ENTITY) -P -Cproject_1/src -Oproject_1/sources.txt

build: $(PROJECT).xpr
$(PROJECT).xpr: project_1/sources.txt
ifdef FPU_IP
	MAX_JOBS=$(JOBS) FPU_IP=project_1/ip TOOL_DIR=$(SCRIPT_DIR) $(VIVADO) -mode batch -source $(SRC_DIR)/project.tcl -tclargs $(TOP_LEVEL_ENTITY) $(DEVICE) project_1/sources.txt $(SRC_DIR)/project.xdc
else
	MAX_JOBS=$(JOBS) TOOL_DIR=$(SCRIPT_DIR) $(VIVADO) -mode batch -source $(SRC_DIR)/project.tcl -tclargs $(TOP_LEVEL_ENTITY) $(DEVICE) project_1/sources.txt $(SRC_DIR)/project.xdc
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