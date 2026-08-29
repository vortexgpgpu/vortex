ROOT_DIR := $(realpath ../../../..)
include $(ROOT_DIR)/config.mk
# exports VERILATOR/VERILATOR_PATH, which gen_sources.sh resolves from the
# environment rather than PATH
include $(VORTEX_HOME)/hw/syn/common.mk

ifneq ($(findstring Makefile, $(MAKEFILE_LIST)), Makefile)
help:
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<hw|sim>"
	$(ECHO) "      Package the AFU as IP-XACT and link a V80 device binary."
	$(ECHO) ""
	$(ECHO) "  make clean"
	$(ECHO) "      Remove generated files."
	$(ECHO) ""
endif

TARGET ?= hw

PREFIX ?= build$(XLEN)

MAX_JOBS ?= 8

# V80 device. Override for a different speed grade or package.
DEVICE ?= xcv80-lsva4737-2MHP-e-S

SRC_DIR := $(VORTEX_HOME)/hw/syn/xilinx/aved

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi
AFU_DIR := $(RTL_DIR)/afu/aved
AFU_COMMON_DIR := $(RTL_DIR)/afu/common
SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts

VIVADO := $(XILINX_VIVADO)/bin/vivado
SLASHKIT ?= slashkit

CP = cp -rf
RMDIR = rm -rf
ECHO = @echo

NCPUS := $(shell lscpu | grep "^Core(s) per socket:" | awk '{print $$4}')
JOBS ?= $(shell echo $$(( $(NCPUS) > $(MAX_JOBS) ? $(MAX_JOBS) : $(NCPUS) )))

BUILD_DIR := $(PREFIX)_aved_$(TARGET)
BIN_DIR := $(BUILD_DIR)/bin

IP_DIR := $(BUILD_DIR)/vortex_afu_ip
VBIN_CONTAINER := $(BIN_DIR)/vortex_afu.vbin

# RTL kernels are only supported by the hardware and simulation platforms.
# The emulation platform is a behavioural C-model that does not accept them.
ifeq ($(TARGET), emu)
$(error TARGET=emu is unavailable: the V80 emulation platform is a C-model \
        that does not accept RTL kernels. Use TARGET=avedsim for fast \
        functional iteration, or --driver=simx for the C-model)
endif

ifneq ($(TARGET),$(findstring $(TARGET), hw sim))
$(error TARGET must be hw or sim (got '$(TARGET)'))
endif

# Control RTL debug tracing states
DBG_TRACE_FLAGS += -DDBG_TRACE_PIPELINE
DBG_TRACE_FLAGS += -DDBG_TRACE_MEM
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE
DBG_TRACE_FLAGS += -DDBG_TRACE_AFU
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
CONFIGS_8c	:= -DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=8
CONFIGS_16c	:= -DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=16
CONFIGS_32c := -DVX_CFG_NUM_CLUSTERS=2 -DVX_CFG_NUM_CORES=16
CONFIGS_64c := -DVX_CFG_NUM_CLUSTERS=4 -DVX_CFG_NUM_CORES=16
override CONFIGS += $(CONFIGS_$(NUM_CORES)c)
endif

# `override`, as in platforms.mk: a command-line `make CONFIGS=...` otherwise
# discards these appends, and a build silently missing -DSYNTHESIS elaborates
# the simulation-only paths.
override CONFIGS += -DSYNTHESIS -DVIVADO

ifeq ($(TARGET), sim)
	override CONFIGS += -DSIMULATION
endif

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

# Per-extension RTL sources — single source of truth, appended based on
# the resolved XCONFIGS.
include $(VORTEX_HOME)/hw/syn/extensions.mk

# load platform settings
include $(SRC_DIR)/platforms.mk

# Memory connectivity tags resolved into the linker config.
MEM_TAG ?= HBM0
HOST_TAG ?= HOST

# HOST routes m_axi_host to the QDMA slave bridge, whose reads never complete
# on the compute shell: the CP hangs on its first ring fetch and reports
# nothing -- the queue reads back armed and correct while head stays at 0. That
# is a full synthesis run to discover, and it has already cost one, so refuse
# the tag at parse time rather than emitting the config. See platforms.mk.
ifeq ($(HOST_TAG),HOST)
$(error HOST_TAG=HOST routes m_axi_host to the QDMA slave bridge, whose reads never complete on the compute shell; the CP would hang on its first ring fetch. Use HOST_TAG=HBM1 (the platforms.mk default))
endif

# platforms.mk states the kernel clock in MHz; the linker config wants Hz.
KERNEL_FREQ_HZ := $(shell echo $$(( $(KERNEL_FREQ) * 1000000 )))

# Debugging
ifdef DEBUG
	CFLAGS += -DVX_DBG_DEBUG_LEVEL=$(DEBUG) $(DBG_TRACE_FLAGS)
else
	CFLAGS += -DNDEBUG
endif

# Enable scope analyzer
ifdef SCOPE
	CFLAGS += -DSCOPE $(DBG_SCOPE_FLAGS)
	SCOPE_JSON += $(BIN_DIR)/scope.json
endif

# Enable perf counters
ifdef PERF
	CFLAGS += -DPERF_ENABLE
endif

# compilation flags
CFLAGS += -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN)
# raw CONFIGS carries only an enum *value* (e.g. VX_CFG_TCU_TYPE=TFR); append the
# resolved type selectors so an enum override selects its RTL implementation
# (without this, VX_tcu_fedp elaborates 'no_implementation_selected').
CFLAGS += $(CONFIGS) $(filter -DVX_CFG_TCU_TYPE_% -DVX_CFG_FPU_TYPE_%,$(XCONFIGS))
CFLAGS += $(RTL_INCLUDE)

# ast dump flags
XML_CFLAGS = $(filter-out -DSYNTHESIS -DVIVADO, $(CFLAGS)) $(RTL_PKGS) -I$(DPI_DIR) -DSV_DPI

.PHONY: all clean gen-sources gen-ast scope-json package-ip gen-bin check-axil-blocks

# Fail before synthesis if any 16-byte block in the CP AXI-Lite regfile is only
# partially decoded. Measured on V80 silicon: such a block DECERRs on EVERY
# word, including the words that ARE implemented -- the shell -> s_axi_control
# path resolves at 128-bit granularity and rejects a block it cannot fully
# resolve, so the per-word decode never gets consulted. No simulator reproduces
# it; the symptom is a register reading 0xFFFFFFFF on hardware for no reason
# visible in the RTL. Finding that cost ten days. A synthesis run costs ~37
# minutes, so this check pays for itself the first time it fires.
check-axil-blocks:
	@python3 $(SCRIPT_DIR)/check_axil_blocks.py $(RTL_DIR)/cp/VX_cp_axil_regfile.sv

all: check-axil-blocks $(VBIN_CONTAINER)

gen-sources: $(BUILD_DIR)/sources.txt
$(BUILD_DIR)/sources.txt:
	mkdir -p $(BUILD_DIR); cd $(BUILD_DIR); $(SCRIPT_DIR)/gen_sources.sh -P $(CFLAGS) -Csrc -Osources.txt

gen-ast: $(BUILD_DIR)/vortex.xml
$(BUILD_DIR)/vortex.xml:
	mkdir -p $(BUILD_DIR); cd $(BUILD_DIR); $(VERILATOR_PATH)/bin/verilator --xml-only -O0 $(XML_CFLAGS) vortex_afu.v --xml-output vortex.xml

scope-json: $(BIN_DIR)/scope.json
$(BIN_DIR)/scope.json: $(BUILD_DIR)/vortex.xml
	mkdir -p $(BIN_DIR); cd $(BUILD_DIR); $(SCRIPT_DIR)/scope.py vortex.xml -o bin/scope.json

$(BUILD_DIR)/config.cfg: $(SRC_DIR)/config.cfg.tmpl
	mkdir -p $(BUILD_DIR)
	sed -e 's|@MEM_TAG@|$(MEM_TAG)|g' -e 's|@HOST_TAG@|$(HOST_TAG)|g' \
	    -e 's|@KERNEL_FREQ_HZ@|$(KERNEL_FREQ_HZ)|g' $< > $@
	@grep -q '@[A-Z_]*@' $@ && { echo "ERROR: unsubstituted placeholder in $@"; grep -n '@[A-Z_]*@' $@; rm -f $@; exit 1; } || true

package-ip: $(IP_DIR)/component.xml
$(IP_DIR)/component.xml: $(BUILD_DIR)/sources.txt
	cd $(BUILD_DIR); TOOL_DIR=$(SCRIPT_DIR) $(VIVADO) -mode batch -source $(SRC_DIR)/package_ip.tcl -tclargs vortex_afu sources.txt . $(DEVICE)

gen-bin: $(VBIN_CONTAINER)
# TOOL_DIR is consumed by pre_opt_hook.tcl to locate hw/scripts. The pre-synth
# hook is the linker's only opportunity to set per-step properties on impl_1,
# and it is what attaches the pre-opt hook that resolves the async-BRAM
# placeholders.
$(VBIN_CONTAINER): $(IP_DIR)/component.xml $(BUILD_DIR)/config.cfg $(SCOPE_JSON)
	mkdir -p $(BIN_DIR)
	TOOL_DIR=$(SCRIPT_DIR) $(SLASHKIT) link -c $(BUILD_DIR)/config.cfg -p $(TARGET) -o $(VBIN_CONTAINER) \
	  -k $(IP_DIR)/component.xml --vivado $(VIVADO) \
	  --pre-synth-tcls $(SRC_DIR)/pre_synth_hook.tcl

clean:
	$(RMDIR) $(BUILD_DIR)
