ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

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
CONFIGS_1c  := -DNUM_CLUSTERS=1 -DNUM_CORES=1
CONFIGS_2c  := -DNUM_CLUSTERS=1 -DNUM_CORES=2
CONFIGS_4c  := -DNUM_CLUSTERS=1 -DNUM_CORES=4  -DL2_ENABLE
CONFIGS_8c	:= -DNUM_CLUSTERS=1 -DNUM_CORES=8  -DL2_ENABLE
CONFIGS_16c	:= -DNUM_CLUSTERS=1 -DNUM_CORES=16 -DL2_ENABLE
CONFIGS_32c := -DNUM_CLUSTERS=2 -DNUM_CORES=16 -DL2_ENABLE
CONFIGS_64c := -DNUM_CLUSTERS=4 -DNUM_CORES=16 -DL2_ENABLE
CONFIGS += $(CONFIGS_$(NUM_CORES)c)
endif

# include paths
FPU_INCLUDE = -I$(RTL_DIR)/fpu
ifeq (,$(filter -DEXT_F_DISABLE, $(CONFIGS)))
	FPU_INCLUDE += -I$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -I$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -I$(THIRD_PARTY_DIR)/cvfpu/src/fpu_div_sqrt_mvp/hdl -I$(THIRD_PARTY_DIR)/cvfpu/src
endif
RTL_INCLUDE = -I$(ROOT_DIR)/hw -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache
RTL_INCLUDE += $(FPU_INCLUDE)

RTL_INCLUDE += -I$(LIB_DIR)/no_mem

# Add TCU extension sources
ifneq (,$(filter -DEXT_TCU_ENABLE, $(CONFIGS)))
	RTL_PKGS += $(RTL_DIR)/tcu/VX_tcu_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/tcu
	ifneq (,$(filter -DTCU_TYPE_DPI, $(CONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/dpi
	endif
	ifneq (,$(filter -DTCU_TYPE_DSP, $(CONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/dsp
	endif
	ifneq (,$(filter -DTCU_TYPE_BHF, $(CONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/bhf
		RTL_INCLUDE += -J$(THIRD_PARTY_DIR)/hardfloat/source/RISCV
		RTL_INCLUDE += -I$(THIRD_PARTY_DIR)/hardfloat/source
	endif
	ifneq (,$(filter -DTCU_TYPE_TFR, $(CONFIGS)))
		RTL_INCLUDE += -I$(RTL_DIR)/tcu/tfr
	endif
endif

# Add V extension sources
ifneq (,$(filter -DEXT_V_ENABLE, $(CONFIGS)))
	RTL_PKGS += $(RTL_DIR)/vpu/VX_vpu_pkg.sv
	RTL_INCLUDE += -I$(RTL_DIR)/vpu
endif

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

CFLAGS += -DSYNTHESIS -DASIC -DYOSYS
CFLAGS += -DXLEN_$(XLEN)
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
	# Remove lines starting $finish or $display from generated source
	sed -ri '/^[[:space:]]*[$$](finish|display)/d' $(BUILD_DIR)/project.v

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
	$(SYN_LIB_ENV) \
	TOOL_DIR=$(SCRIPT_DIR) \
	RUN_MAP=1 \
	$(SRC_DIR)/run_synth.sh

timing: $(BUILD_DIR)/sources.txt
	cd $(BUILD_DIR); TOP=$(TOP_LEVEL_ENTITY) \
	SRC_FILE=sources.txt \
	SDC_FILE=$(SRC_DIR)/project.sdc \
	BB_MODULES="VX_dp_ram_asic,VX_sp_ram_asic" \
	$(SYN_LIB_ENV) \
	TOOL_DIR=$(SCRIPT_DIR) \
	RUN_STA=1 \
	$(SRC_DIR)/run_synth.sh

clean:
	$(RMDIR) $(BUILD_DIR)
