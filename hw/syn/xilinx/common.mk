# Shared by the Xilinx flows (xrt, aved): configuration, RTL sources and the
# source-generation rules. The including flow sets BUILD_DIR, BIN_DIR, AFU_DIR
# and any TARGET-specific CONFIGS before including this, and includes
# platform.mk after it.

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi
AFU_COMMON_DIR := $(RTL_DIR)/afu/common
SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts

VIVADO := $(XILINX_VIVADO)/bin/vivado

CP = cp -rf
RMDIR = rm -rf
ECHO = @echo

MAX_JOBS ?= 8
NCPUS := $(shell lscpu | grep "^Core(s) per socket:" | awk '{print $$4}')
JOBS ?= $(shell echo $$(( $(NCPUS) > $(MAX_JOBS) ? $(MAX_JOBS) : $(NCPUS) )))

# Control RTL debug tracing states
DBG_TRACE_FLAGS += -DDBG_TRACE_PIPELINE
DBG_TRACE_FLAGS += -DDBG_TRACE_MEM
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE
DBG_TRACE_FLAGS += -DDBG_TRACE_AFU
DBG_TRACE_FLAGS += -DDBG_TRACE_TEX
DBG_TRACE_FLAGS += -DDBG_TRACE_RASTER
DBG_TRACE_FLAGS += -DDBG_TRACE_OM
DBG_TRACE_FLAGS += -DDBG_TRACE_GBAR
DBG_TRACE_FLAGS += -DDBG_TRACE_TCU

# Control logic analyzer monitors
DBG_SCOPE_FLAGS += -DDBG_SCOPE_AFU
DBG_SCOPE_FLAGS += -DDBG_SCOPE_ISSUE
DBG_SCOPE_FLAGS += -DDBG_SCOPE_FETCH
DBG_SCOPE_FLAGS += -DDBG_SCOPE_LSU

ifdef NUM_CORES
# cluster configuration
CONFIGS_1c  := -DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=1
CONFIGS_2c  := -DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=2
CONFIGS_4c  := -DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=4
CONFIGS_8c  := -DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=8
CONFIGS_16c := -DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=16
CONFIGS_32c := -DVX_CFG_NUM_CLUSTERS=2 -DVX_CFG_NUM_CORES=16
CONFIGS_64c := -DVX_CFG_NUM_CLUSTERS=4 -DVX_CFG_NUM_CORES=16
override CONFIGS += $(CONFIGS_$(NUM_CORES)c)
endif

# `override` throughout: a command-line `make CONFIGS=...` otherwise discards
# these appends, and a build silently missing -DSYNTHESIS elaborates the
# simulation-only paths.
override CONFIGS += -DSYNTHESIS -DVIVADO

XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py --config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS) -DVX_CFG_XLEN=$(XLEN)')

# include sources
RTL_PKGS = $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/fpu/VX_fpu_pkg.sv
FPU_INCLUDE = -I$(RTL_DIR)/fpu
ifneq (,$(filter -DVX_CFG_FPU_TYPE_FPNEW, $(XCONFIGS)))
	RTL_PKGS += $(THIRD_PARTY_DIR)/cvfpu/src/fpnew_pkg.sv $(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src/cf_math_pkg.sv $(THIRD_PARTY_DIR)/cvfpu/src/fpu_div_sqrt_mvp/hdl/defs_div_sqrt_mvp.sv
	FPU_INCLUDE += -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -J$(THIRD_PARTY_DIR)/cvfpu/src/fpu_div_sqrt_mvp/hdl -J$(THIRD_PARTY_DIR)/cvfpu/src
endif
RTL_INCLUDE = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/cache -I$(AFU_DIR) -I$(AFU_COMMON_DIR)
RTL_INCLUDE += $(FPU_INCLUDE)

# Command Processor sources. RTL_PKGS holds packages only; the CP
# interface files are auto-discovered via the -I path below.
RTL_PKGS += $(RTL_DIR)/cp/VX_cp_pkg.sv
RTL_INCLUDE += -I$(RTL_DIR)/cp

# Per-extension RTL sources (TCU/DXA/V/RTU/TEX/RASTER/OM) —
# single source of truth, appended based on the resolved XCONFIGS.
include $(VORTEX_HOME)/hw/syn/extensions.mk

# Enable perf counters
ifdef PERF
	CFLAGS += -DPERF_ENABLE
endif

# Enable scope analyzer
ifdef SCOPE
	CFLAGS += -DSCOPE $(DBG_SCOPE_FLAGS)
	SCOPE_JSON += $(BIN_DIR)/scope.json
endif

# compilation flags (the flow adds its DEBUG/NDEBUG flags)
COMPILE_CFLAGS = -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN)
# raw CONFIGS carries only an enum *value* (e.g. VX_CFG_TCU_TYPE=TFR); append the
# resolved type selectors so an enum override selects its RTL implementation
# (without this, VX_tcu_fedp elaborates 'no_implementation_selected').
COMPILE_CFLAGS += $(CONFIGS) $(filter -DVX_CFG_TCU_TYPE_% -DVX_CFG_FPU_TYPE_%,$(XCONFIGS))
COMPILE_CFLAGS += $(RTL_INCLUDE)

# ast dump flags
XML_CFLAGS = $(filter-out -DSYNTHESIS -DVIVADO, $(CFLAGS)) $(RTL_PKGS) -I$(DPI_DIR) -DSV_DPI

# Fail before synthesis if any 16-byte block in the CP AXI-Lite regfile is only
# partially decoded, on cards whose shell resolves AXI-Lite in blocks (the
# descriptor's axil_block_bytes). Measured on V80 silicon: such a block DECERRs
# on EVERY word, including the words that ARE implemented. No simulator
# reproduces it; the symptom is a register reading 0xFFFFFFFF on hardware for no
# reason visible in the RTL.
check-axil-blocks:
	$(if $(filter-out 0,$(PLATFORM_AXIL_BLOCK_BYTES)),@python3 $(SCRIPT_DIR)/check_axil_blocks.py $(RTL_DIR)/cp/VX_cp_axil_regfile.sv)

# The sources are elaborated by CIRCT into the one netlist Vivado reads
# (gen_sources.sh -E); the Xilinx floating-point cores stay instances, bound to
# the cores xilinx_ip_gen.tcl generates.
gen-sources: $(BUILD_DIR)/sources.txt
$(BUILD_DIR)/sources.txt:
	mkdir -p $(BUILD_DIR); cd $(BUILD_DIR); $(SCRIPT_DIR)/gen_sources.sh -P $(CFLAGS) -Csrc -Osources.txt \
		-Tvortex_afu -E -X$(SCRIPT_DIR)/xilinx_ip_stubs.sv

gen-ast: $(BUILD_DIR)/vortex.xml
$(BUILD_DIR)/vortex.xml:
	mkdir -p $(BUILD_DIR); cd $(BUILD_DIR); $(VERILATOR_PATH)/bin/verilator --xml-only -O0 $(XML_CFLAGS) vortex_afu.v --xml-output vortex.xml

scope-json: $(BIN_DIR)/scope.json
$(BIN_DIR)/scope.json: $(BUILD_DIR)/vortex.xml
	mkdir -p $(BIN_DIR); cd $(BUILD_DIR); $(SCRIPT_DIR)/scope.py vortex.xml -o bin/scope.json

.PHONY: check-axil-blocks gen-sources gen-ast scope-json
