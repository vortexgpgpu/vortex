ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

SRC_DIR := $(VORTEX_HOME)/hw/syn/yosys

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
ifneq (,$(findstring FPU_FPNEW,$(CONFIGS)))
	FPU_INCLUDE += -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -J$(THIRD_PARTY_DIR)/cvfpu/src/fpu_div_sqrt_mvp/hdl -J$(THIRD_PARTY_DIR)/cvfpu/src
endif
RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache
RTL_INCLUDE += $(FPU_INCLUDE)

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

CFLAGS += -DSYNTHESIS -DYOSYS
CFLAGS += -DXLEN_$(XLEN)
CFLAGS += $(CONFIGS)
CFLAGS += $(RTL_INCLUDE)

# Build targets
all: clean build

gen-sources: $(BUILD_DIR)/sources.txt
$(BUILD_DIR)/sources.txt:
	mkdir -p $(BUILD_DIR); cd $(BUILD_DIR); $(SCRIPT_DIR)/gen_sources.sh -P $(CFLAGS) -Csrc -Osources.txt

sv2v: $(BUILD_DIR)/project.v
$(BUILD_DIR)/project.v: gen-sources
	cd $(BUILD_DIR); $(SCRIPT_DIR)/sv2v.sh -t$(TOP_LEVEL_ENTITY) -Isrc -oproject.v

build: $(BUILD_DIR)/project.v
	cd $(BUILD_DIR); $(SRC_DIR)/synth.sh -t$(TOP_LEVEL_ENTITY) -sproject.v

elaborate: $(BUILD_DIR)/project.v
	cd $(BUILD_DIR); $(SRC_DIR)/synth.sh -t$(TOP_LEVEL_ENTITY) -sproject.v -P="elaborate"

synthesis: $(BUILD_DIR)/project.v
	cd $(BUILD_DIR); $(SRC_DIR)/synth.sh -t$(TOP_LEVEL_ENTITY) -sproject.v -P="synthesis"

clean:
	$(RMDIR) $(BUILD_DIR)
