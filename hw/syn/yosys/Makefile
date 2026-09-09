ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk
include $(ROOT_DIR)/hw/syn/common.mk    # SV2V_PATH/YOSYS_PATH/STA_PATH + exported SV2V/YOSYS/STA

DELAY_UNC 	?= 0.02
DELAY_IO  	?= 0.05
ABC_DRIVER_CELL ?=
ABC_LOAD        ?=
YOSYS_FLATTEN   ?= 1
YOSYS_SHARE     ?= 1
SAIF_FILE 	?=
SAIF_INST 	?=
PDK         ?= asap7
ASAP7_VT    ?= rvt
CORNER      ?= tt
ifneq ($(CORNER),$(filter $(CORNER),tt ss ff))
$(error Unsupported CORNER=$(CORNER); expected tt, ss, or ff)
endif
ifneq ($(ASAP7_VT),$(filter $(ASAP7_VT),lvt rvt))
$(error Unsupported ASAP7_VT=$(ASAP7_VT); expected lvt or rvt)
endif

# Synthesis optimization level (standardized across hw/syn):
#   0 -- fastest compile, 3 -- default. NOT supported on this backend.
OPT_LEVEL ?= 3
ifneq ($(OPT_LEVEL),3)
$(error OPT_LEVEL=$(OPT_LEVEL) requested but not supported on the yosys backend)
endif

SRC_DIR := $(VORTEX_HOME)/hw/syn/yosys
LIB_DIR := $(VORTEX_HOME)/hw/syn/libs
ASAP7_DIR := $(ROOT_DIR)/hw/syn/libs/asap7
ASAP7_INSTALL := $(ASAP7_DIR)/install.sh
ASAP7_LIB := $(ASAP7_DIR)/lib/asap7_$(ASAP7_VT)_$(CORNER).lib
ASAP7_CELL_MODELS := $(ASAP7_DIR)/verilog/asap7_$(ASAP7_VT)_cells.v

ifeq ($(PDK),asap7)
DEFAULT_LIB := $(ASAP7_LIB)
DEFAULT_CLOCK_FREQ := 800
ifeq ($(strip $(ABC_DRIVER_CELL)),)
ABC_DRIVER_CELL := BUFx4_ASAP7_75t_R
endif
ifeq ($(strip $(ABC_LOAD)),)
ABC_LOAD := 5.0
endif
else ifeq ($(PDK),nangate45)
DEFAULT_LIB := $(LIB_DIR)/NangateOpenCellLibrary_typical.lib
DEFAULT_CLOCK_FREQ := 800
else
$(error Unsupported PDK=$(PDK); expected asap7 or nangate45)
endif

# Target clock, defaulted per PDK above. It must bind for Fmax to mean anything:
# a design that closes with slack reports its target back, not its capability.
CLOCK_FREQ ?= $(DEFAULT_CLOCK_FREQ)

ifneq ($(strip $(LIB_TGT)),)
SYN_LIB_ENV := LIB_TGT=$(LIB_TGT)
else ifneq ($(strip $(LIB_ROOT)),)
SYN_LIB_ENV := LIB_ROOT=$(LIB_ROOT)
else
LIB_TGT := $(DEFAULT_LIB)
SYN_LIB_ENV := LIB_TGT=$(DEFAULT_LIB)
endif

TOP_LEVEL_ENTITY ?= Vortex
PREFIX ?= build
SDC_FILE ?= $(SRC_DIR)/project.sdc
DUT_FILELIST ?=

TB_TOP ?= tb
TB_FILELIST ?=
SIM_ARGS ?=
GATE_SIM_FLAGS ?=
GATE_DUT_INSTANCE ?= dut
GATE_SAIF_RAW ?= gate_raw.saif
GATE_SAIF ?= $(abspath $(BUILD_DIR)/gate.saif)

ifneq ($(strip $(DUT_FILELIST)),)
BB_MODULES ?=
else
BB_MODULES ?= VX_dp_ram_asic,VX_sp_ram_asic
endif

SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts
RTL_DIR := $(VORTEX_HOME)/hw/rtl

CP = cp -rf
RMDIR = rm -rf
ECHO = @echo

BUILD_DIR := $(PREFIX)_$(TOP_LEVEL_ENTITY)
BIN_DIR := $(BUILD_DIR)/bin

# control RTL debug tracing states
DBG_TRACE_FLAGS += -DDBG_TRACE_PIPELINE
DBG_TRACE_FLAGS += -DDBG_TRACE_MEM
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE
DBG_TRACE_FLAGS += -DDBG_TRACE_AFU
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
CONFIGS_4c  := -DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=4  -DVX_CFG_L2_ENABLE
CONFIGS_8c	:= -DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=8  -DVX_CFG_L2_ENABLE
CONFIGS_16c	:= -DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=16 -DVX_CFG_L2_ENABLE
CONFIGS_32c := -DVX_CFG_NUM_CLUSTERS=2 -DVX_CFG_NUM_CORES=16 -DVX_CFG_L2_ENABLE
CONFIGS_64c := -DVX_CFG_NUM_CLUSTERS=4 -DVX_CFG_NUM_CORES=16 -DVX_CFG_L2_ENABLE
CONFIGS += $(CONFIGS_$(NUM_CORES)c)
endif

# `override`, because these are facts about this flow, not preferences: without
# ASIC the RAM macros infer to flip-flops (a cache DUT lands at a million cells
# and zero SRAM area) and without SYNTHESIS the simulation-only constructs come
# along. A caller that sets CONFIGS on the MAKE COMMAND LINE would otherwise kill
# every `CONFIGS +=` in this file silently -- command-line variables win over
# `+=`, and nothing warns. Extra defines belong in EXTRA_CONFIGS below.
override CONFIGS += -DSYNTHESIS -DASIC -DYOSYS

# Caller-supplied extra design macros (e.g. enabling extensions on a stock top).
# Folded in before XCONFIGS so the conditional include blocks below pick them up.
override CONFIGS += $(EXTRA_CONFIGS)

XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py --config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS) -DVX_CFG_XLEN=$(XLEN)')

# include paths
RTL_INCLUDE = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/vm -I$(RTL_DIR)/cache
ifneq (,$(filter -DVX_CFG_EXT_F_ENABLE, $(XCONFIGS)))
	RTL_INCLUDE += -I$(RTL_DIR)/fpu
	ifneq (,$(filter -DVX_CFG_FPU_TYPE_FPNEW, $(XCONFIGS)))
		RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/cvfpu/src
		RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include
		RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src
		RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/cvfpu/src/fpu_div_sqrt_mvp/hdl
	endif
endif

RTL_INCLUDE += -I$(LIB_DIR)/no_mem

# Per-extension RTL sources (TCU/DXA/V/RTU/TEX/RASTER/OM) —
# single source of truth, appended based on the resolved XCONFIGS.
include $(VORTEX_HOME)/hw/syn/extensions.mk

# Caller-supplied extra include paths (e.g. a unittest top + its deps).
RTL_INCLUDE += $(EXTRA_INCLUDE)

# Debugging
ifdef DEBUG
	CFLAGS += $(DBG_TRACE_FLAGS)
else
	CFLAGS += -DNDEBUG
endif

# Enable scope analyzer
ifdef SCOPE
	CFLAGS += -DSCOPE $(DBG_SCOPE_FLAGS)
	SCOPE_JSON += $(BUILD_DIR)/scope.json
endif

# Enable perf counters
ifdef PERF
	CFLAGS += -DPERF_ENABLE
endif

CFLAGS += -DVX_CFG_XLEN=$(XLEN) -DVX_CFG_XLEN_$(XLEN)
# raw CONFIGS carries only an enum *value* (e.g. VX_CFG_TCU_TYPE=TFR); append the
# resolved type selectors so an enum override selects its RTL implementation.
CFLAGS += $(CONFIGS) $(filter -DVX_CFG_TCU_TYPE_% -DVX_CFG_FPU_TYPE_%,$(XCONFIGS))
CFLAGS += $(RTL_INCLUDE)

# Build targets
all: timing

.PHONY: all synthesis techmap timing gate-sim gate-saif normalize-saif asap7 asap7-models clean

asap7:
	$(ASAP7_INSTALL) --install-dir=$(ASAP7_DIR) --vt=$(ASAP7_VT)

asap7-models: asap7
	$(ASAP7_INSTALL) --install-dir=$(ASAP7_DIR) --vt=$(ASAP7_VT) --models

gen-sources: $(BUILD_DIR)/src
$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/src: | $(BUILD_DIR)
	$(SCRIPT_DIR)/gen_sources.sh -P $(CFLAGS) -C$(BUILD_DIR)/src

sv2v: $(BUILD_DIR)/project.v
$(BUILD_DIR)/project.v: $(BUILD_DIR)/src
	$(SCRIPT_DIR)/sv2v.sh -t$(TOP_LEVEL_ENTITY) -I$(BUILD_DIR)/src -o$(BUILD_DIR)/project.v
	# Remove severity/control system-task lines yosys can't synthesize
	# ($fatal/$error/$warning/$info/$stop are used by the cvfpu sources).
	sed -ri '/^[[:space:]]*[$$](finish|display|fatal|error|warning|info|stop)/d' $(BUILD_DIR)/project.v

gen-sources: $(BUILD_DIR)/sources.txt
$(BUILD_DIR)/sources.txt: $(BUILD_DIR)/project.v
	echo $(realpath $(BUILD_DIR)/project.v) > $(BUILD_DIR)/sources.txt

ifneq ($(strip $(DUT_FILELIST)),)
SYN_SRC_FILE := $(abspath $(DUT_FILELIST))
SYN_SRC_DEPS := $(DUT_FILELIST)
else
SYN_SRC_FILE := sources.txt
SYN_SRC_DEPS := $(BUILD_DIR)/sources.txt
endif

ifeq ($(PDK),asap7)
SYN_LIB_DEPS := asap7
endif

synthesis: $(SYN_SRC_DEPS) $(SYN_LIB_DEPS) | $(BUILD_DIR)
	cd $(BUILD_DIR); TOP=$(TOP_LEVEL_ENTITY) \
	SRC_FILE=$(SYN_SRC_FILE) \
	SDC_FILE=$(SDC_FILE) \
	BB_MODULES="$(BB_MODULES)" \
	PDK=$(PDK) \
	$(SYN_LIB_ENV) \
	TOOL_DIR=$(SCRIPT_DIR) \
	RUN_SYNTH=1 \
	$(SRC_DIR)/run_synth.sh

techmap: $(SYN_SRC_DEPS) $(SYN_LIB_DEPS) | $(BUILD_DIR)
	cd $(BUILD_DIR); TOP=$(TOP_LEVEL_ENTITY) \
	SRC_FILE=$(SYN_SRC_FILE) \
	SDC_FILE=$(SDC_FILE) \
	BB_MODULES="$(BB_MODULES)" \
	CLOCK_FREQ=$(CLOCK_FREQ) \
	ABC_DRIVER_CELL=$(ABC_DRIVER_CELL) \
	ABC_LOAD=$(ABC_LOAD) \
	YOSYS_FLATTEN=$(YOSYS_FLATTEN) \
	YOSYS_SHARE=$(YOSYS_SHARE) \
	PDK=$(PDK) \
	$(SYN_LIB_ENV) \
	TOOL_DIR=$(SCRIPT_DIR) \
	RUN_MAP=1 \
	$(SRC_DIR)/run_synth.sh

timing: $(SYN_SRC_DEPS) $(SYN_LIB_DEPS) | $(BUILD_DIR)
	cd $(BUILD_DIR); TOP=$(TOP_LEVEL_ENTITY) \
	SRC_FILE=$(SYN_SRC_FILE) \
	SDC_FILE=$(SDC_FILE) \
	BB_MODULES="$(BB_MODULES)" \
	CLOCK_FREQ=$(CLOCK_FREQ) \
	DELAY_UNC=$(DELAY_UNC) \
	DELAY_IO=$(DELAY_IO) \
	ABC_DRIVER_CELL=$(ABC_DRIVER_CELL) \
	ABC_LOAD=$(ABC_LOAD) \
	YOSYS_FLATTEN=$(YOSYS_FLATTEN) \
	YOSYS_SHARE=$(YOSYS_SHARE) \
	SAIF_FILE="$(SAIF_FILE)" \
	SAIF_INST="$(SAIF_INST)" \
	PDK=$(PDK) \
	$(SYN_LIB_ENV) \
	TOOL_DIR=$(SCRIPT_DIR) \
	RUN_STA=1 \
	$(SRC_DIR)/run_synth.sh
	@$(SRC_DIR)/synth_summary.py --reports $(BUILD_DIR)/reports \
	  --top $(TOP_LEVEL_ENTITY) --clock-mhz $(CLOCK_FREQ)

gate-sim: techmap asap7-models
	@if [ "$(PDK)" != "asap7" ]; then echo "ERROR: gate-sim requires PDK=asap7 functional models" >&2; exit 1; fi
	@if [ -z "$(TB_FILELIST)" ]; then echo "ERROR: TB_FILELIST is required for gate-sim" >&2; exit 1; fi
	mkdir -p $(BUILD_DIR)/gate_obj
	$(VERILATOR) --binary --timing --trace-saif --trace-underscore --trace-depth 99 \
		-Wno-fatal -Wno-SPECIFYIGN \
		--top-module $(TB_TOP) --Mdir $(abspath $(BUILD_DIR)/gate_obj) \
		$(GATE_SIM_FLAGS) $(ASAP7_CELL_MODELS) \
		$(abspath $(BUILD_DIR)/out/$(TOP_LEVEL_ENTITY)_mapped.v) -f $(abspath $(TB_FILELIST))
	cd $(BUILD_DIR); ./gate_obj/V$(TB_TOP) $(SIM_ARGS)
	@test -s $(BUILD_DIR)/$(GATE_SAIF_RAW) || { \
		echo "ERROR: $(BUILD_DIR)/$(GATE_SAIF_RAW) was not produced; the testbench must call \$$dumpfile(\"$(GATE_SAIF_RAW)\") and \$$dumpvars" >&2; exit 1; }

gate-saif: gate-sim
	$(SCRIPT_DIR)/saif_filter.py --instance $(GATE_DUT_INSTANCE) \
		--top $(TOP_LEVEL_ENTITY) --rename-root \
		-o $(GATE_SAIF) $(BUILD_DIR)/$(GATE_SAIF_RAW)
	@echo "Gate-level SAIF: $(GATE_SAIF)"

normalize-saif:
	@if [ -z "$(SAIF_RAW)" ] || [ -z "$(SAIF_INSTANCE)" ] || [ -z "$(SAIF_FILE)" ]; then \
		echo "ERROR: SAIF_RAW, SAIF_INSTANCE, and SAIF_FILE are required" >&2; exit 1; fi
	$(SCRIPT_DIR)/saif_filter.py --instance $(SAIF_INSTANCE) \
		--top $(TOP_LEVEL_ENTITY) --rename-root -o $(SAIF_FILE) $(SAIF_RAW)

clean:
	$(RMDIR) $(BUILD_DIR)
