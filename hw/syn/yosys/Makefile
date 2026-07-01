ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk
include $(ROOT_DIR)/hw/syn/common.mk    # SV2V_PATH/YOSYS_PATH/STA_PATH + exported SV2V/YOSYS/STA

CLOCK_FREQ	?= 800
DELAY_UNC 	?= 0.02
DELAY_IO  	?= 0.05
SAIF_FILE 	?=
SAIF_INST 	?=

# Synthesis optimization level (standardized across hw/syn):
#   0 -- fastest compile, 3 -- default. NOT supported on this backend.
OPT_LEVEL ?= 3
ifneq ($(OPT_LEVEL),3)
$(error OPT_LEVEL=$(OPT_LEVEL) requested but not supported on the yosys backend)
endif

SRC_DIR := $(VORTEX_HOME)/hw/syn/yosys
LIB_DIR := $(VORTEX_HOME)/hw/syn/libs

DEFAULT_LIB := $(LIB_DIR)/NangateOpenCellLibrary_typical.lib
SYN_LIB_ENV := $(if $(LIB_TGT),LIB_TGT=$(LIB_TGT),$(if $(LIB_ROOT),LIB_ROOT=$(LIB_ROOT),LIB_TGT=$(DEFAULT_LIB)))

TOP_LEVEL_ENTITY ?= Vortex
PREFIX ?= build

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

CONFIGS += -DSYNTHESIS -DASIC -DYOSYS

# Caller-supplied extra design macros (e.g. enabling extensions on a stock top).
# Folded in before XCONFIGS so the conditional include blocks below pick them up.
CONFIGS += $(EXTRA_CONFIGS)

XCONFIGS := $(shell python3 $(ROOT_DIR)/ci/gen_config.py --config=$(VORTEX_HOME)/VX_config.toml --cflags='$(CONFIGS) -DVX_CFG_XLEN=$(XLEN)')

# include paths
RTL_INCLUDE = -I$(ROOT_DIR)/sw -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache
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

# Add TCU extension sources
ifneq (,$(filter -DVX_CFG_EXT_TCU_ENABLE, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/tcu/VX_tcu_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/tcu
	ifneq (,$(filter -DVX_CFG_TCU_TYPE_DPI, $(XCONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/dpi
	endif
	ifneq (,$(filter -DVX_CFG_TCU_TYPE_DSP, $(XCONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/dsp
	endif
	ifneq (,$(filter -DVX_CFG_TCU_TYPE_BHF, $(XCONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/bhf
		RTL_INCLUDE += -J$(THIRD_PARTY_DIR)/hardfloat/source/RISCV
		RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/hardfloat/source
	endif
	ifneq (,$(filter -DVX_CFG_TCU_TYPE_TFR, $(XCONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/tfr
	endif
	ifneq (,$(filter -DVX_CFG_TCU_TYPE_TET, $(XCONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/tet
	endif
endif

# Add DXA extension sources
ifneq (,$(filter -DVX_CFG_EXT_DXA_ENABLE, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/dxa/VX_dxa_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/dxa
endif

# Add V extension sources
ifneq (,$(filter -DVX_CFG_EXT_V_ENABLE, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/vpu/VX_vpu_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/vpu
endif

# Add RASTER extension sources
ifneq (,$(filter -DVX_CFG_EXT_RASTER_ENABLE, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/raster/VX_raster_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/raster
endif

# Add TEX extension sources
ifneq (,$(filter -DVX_CFG_EXT_TEX_ENABLE, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/tex/VX_tex_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/tex
endif

# Add OM extension sources
ifneq (,$(filter -DVX_CFG_EXT_OM_ENABLE, $(XCONFIGS)))
	RTL_PKGS += $(RTL_DIR)/om/VX_om_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/om
endif

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
CFLAGS += $(CONFIGS)
CFLAGS += $(RTL_INCLUDE)

# Build targets
all: timing

gen-sources: $(BUILD_DIR)/src
$(BUILD_DIR)/src:
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

synthesis: $(BUILD_DIR)/sources.txt
	cd $(BUILD_DIR); TOP=$(TOP_LEVEL_ENTITY) \
	SRC_FILE=sources.txt \
	SDC_FILE=$(SRC_DIR)/project.sdc \
	BB_MODULES="VX_dp_ram_asic,VX_sp_ram_asic" \
	$(SYN_LIB_ENV) \
	TOOL_DIR=$(SCRIPT_DIR) \
	RUN_SYNTH=1 \
	$(SRC_DIR)/run_synth.sh

techmap: $(BUILD_DIR)/sources.txt
	cd $(BUILD_DIR); TOP=$(TOP_LEVEL_ENTITY) \
	SRC_FILE=sources.txt \
	SDC_FILE=$(SRC_DIR)/project.sdc \
	BB_MODULES="VX_dp_ram_asic,VX_sp_ram_asic" \
	CLOCK_FREQ=$(CLOCK_FREQ) \
	$(SYN_LIB_ENV) \
	TOOL_DIR=$(SCRIPT_DIR) \
	RUN_MAP=1 \
	$(SRC_DIR)/run_synth.sh

timing: $(BUILD_DIR)/sources.txt
	cd $(BUILD_DIR); TOP=$(TOP_LEVEL_ENTITY) \
	SRC_FILE=sources.txt \
	SDC_FILE=$(SRC_DIR)/project.sdc \
	BB_MODULES="VX_dp_ram_asic,VX_sp_ram_asic" \
	CLOCK_FREQ=$(CLOCK_FREQ) \
	DELAY_UNC=$(DELAY_UNC) \
	DELAY_IO=$(DELAY_IO) \
	SAIF_FILE="$(SAIF_FILE)" \
	SAIF_INST="$(SAIF_INST)" \
	$(SYN_LIB_ENV) \
	TOOL_DIR=$(SCRIPT_DIR) \
	RUN_STA=1 \
	$(SRC_DIR)/run_synth.sh

clean:
	$(RMDIR) $(BUILD_DIR)
