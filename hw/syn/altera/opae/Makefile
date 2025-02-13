ROOT_DIR := $(realpath ../../../..)
include $(ROOT_DIR)/config.mk

DEVICE_FAMILY ?= arria10

PREFIX ?= build$(XLEN)
TARGET ?= fpga

SRC_DIR := $(VORTEX_HOME)/hw/syn/altera/opae

RTL_DIR := $(VORTEX_HOME)/hw/rtl
DPI_DIR := $(VORTEX_HOME)/hw/dpi
AFU_DIR := $(RTL_DIR)/afu/opae
SCRIPT_DIR := $(VORTEX_HOME)/hw/scripts

IP_CACHE_DIR := $(ROOT_DIR)/hw/syn/altera/ip_cache/$(DEVICE_FAMILY)

BUILD_DIR := $(PREFIX)_$(DEVICE_FAMILY)_$(TARGET)_$(NUM_CORES)c

ifeq ($(shell which qsub-synth),)
	RUN_SYNTH=$(OPAE_PLATFORM_ROOT)/bin/run.sh > build.log 2>&1 &
else
	RUN_SYNTH=qsub-synth
endif

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

ifeq ($(DEVICE_FAMILY), stratix10)
    CONFIGS += -DALTERA_S10
endif
ifeq ($(DEVICE_FAMILY), arria10)
    CONFIGS += -DALTERA_A10
endif

ifdef NUM_CORES
# cluster configuration
CONFIGS_1c  := -DNUM_CLUSTERS=1 -DNUM_CORES=1
CONFIGS_2c  := -DNUM_CLUSTERS=1 -DNUM_CORES=2
CONFIGS_4c  := -DNUM_CLUSTERS=1 -DNUM_CORES=4
CONFIGS_8c	:= -DNUM_CLUSTERS=1 -DNUM_CORES=8
CONFIGS_16c	:= -DNUM_CLUSTERS=1 -DNUM_CORES=16
CONFIGS_32c := -DNUM_CLUSTERS=2 -DNUM_CORES=16
CONFIGS_64c := -DNUM_CLUSTERS=4 -DNUM_CORES=16
CONFIGS += $(CONFIGS_$(NUM_CORES)c)
endif

# include sources
RTL_PKGS = $(AFU_DIR)/local_mem_cfg_pkg.sv $(AFU_DIR)/ccip/ccip_if_pkg.sv
RTL_PKGS += $(RTL_DIR)/VX_gpu_pkg.sv $(RTL_DIR)/fpu/VX_fpu_pkg.sv
FPU_INCLUDE = -I$(RTL_DIR)/fpu
ifneq (,$(findstring FPU_FPNEW,$(CONFIGS)))
	RTL_PKGS += $(THIRD_PARTY_DIR)/cvfpu/src/fpnew_pkg.sv $(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src/cf_math_pkg $(THIRD_PARTY_DIR)/cvfpu/src/fpu_div_sqrt_mvp/hdl/defs_div_sqrt_mvp.sv
	FPU_INCLUDE += -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/include -J$(THIRD_PARTY_DIR)/cvfpu/src/common_cells/src -J$(THIRD_PARTY_DIR)/cvfpu/src/fpu_div_sqrt_mvp/hdl -J$(THIRD_PARTY_DIR)/cvfpu/src
endif
RTL_INCLUDE = -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/mem -I$(RTL_DIR)/cache -I$(AFU_DIR) -I$(IP_CACHE_DIR)
RTL_INCLUDE += $(FPU_INCLUDE)

# compilation flags
CFLAGS += -DSYNTHESIS -DQUARTUS
CFLAGS += -DXLEN_$(XLEN)
CFLAGS += $(CONFIGS)
CFLAGS += $(RTL_INCLUDE)

ifneq ($(TARGET), fpga)
	CFLAGS += -DSIMULATION
endif

# Debugging
ifdef DEBUG
	ifneq ($(TARGET), fpga)
		CFLAGS += -DDEBUG_LEVEL=$(DEBUG) $(DBG_TRACE_FLAGS)
	else
		CFLAGS += -DNDEBUG
	endif
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

# ast dump flags
XML_CFLAGS = $(filter-out -DSYNTHESIS -DQUARTUS, $(CFLAGS)) $(RTL_PKGS) -I$(AFU_DIR)/ccip -I$(DPI_DIR) -DPLATFORM_PROVIDES_LOCAL_MEMORY -DPLATFORM_MEMORY_NUM_BANKS=1 -DNOPAE -DSV_DPI

all: swconfig ip-gen setup build

ip-gen: $(IP_CACHE_DIR)/ip-gen.log
$(IP_CACHE_DIR)/ip-gen.log:
	$(SCRIPT_DIR)/altera_ip_gen.sh $(IP_CACHE_DIR)

swconfig: vortex_afu.h
vortex_afu.h: $(SRC_DIR)/vortex_afu.json
	afu_json_mgr json-info --afu-json=$^ --c-hdr=$@

$(BUILD_DIR)/setup.cfg:
	mkdir -p $(BUILD_DIR); cp $(SRC_DIR)/setup.cfg $(BUILD_DIR)/setup.cfg

$(BUILD_DIR)/vortex_afu.qsf:
	mkdir -p $(BUILD_DIR); cp $(SRC_DIR)/vortex_afu.qsf $(BUILD_DIR)/vortex_afu.qsf

$(BUILD_DIR)/vortex_afu.json:
	mkdir -p $(BUILD_DIR); cp $(SRC_DIR)/vortex_afu.json $(BUILD_DIR)/vortex_afu.json

gen-sources: $(BUILD_DIR)/sources.txt
$(BUILD_DIR)/sources.txt:
	mkdir -p $(BUILD_DIR); $(SCRIPT_DIR)/gen_sources.sh $(CFLAGS) -C$(BUILD_DIR)/src -O$(BUILD_DIR)/sources.txt

setup: $(BUILD_DIR)/synth
$(BUILD_DIR)/synth: $(BUILD_DIR)/sources.txt $(BUILD_DIR)/setup.cfg $(BUILD_DIR)/vortex_afu.qsf $(BUILD_DIR)/vortex_afu.json
ifeq ($(TARGET), asesim)
	afu_sim_setup -s $(BUILD_DIR)/setup.cfg $(BUILD_DIR)/synth
else
	afu_synth_setup -s $(BUILD_DIR)/setup.cfg $(BUILD_DIR)/synth
endif

build: ip-gen setup $(SCOPE_JSON)
ifeq ($(TARGET), asesim)
	make -C $(BUILD_DIR)/synth > $(BUILD_DIR)/synth/build.log 2>&1 &
else
	cd $(BUILD_DIR)/synth && $(RUN_SYNTH)
endif

gen-ast: $(BUILD_DIR)/vortex.xml
$(BUILD_DIR)/vortex.xml: setup
	verilator --xml-only -O0 $(XML_CFLAGS) vortex_afu.sv --xml-output $(BUILD_DIR)/vortex.xml

scope-json: $(BUILD_DIR)/scope.json
$(BUILD_DIR)/scope.json: $(BUILD_DIR)/vortex.xml
	$(SCRIPT_DIR)/scope.py $(BUILD_DIR)/vortex.xml -o $(BUILD_DIR)/scope.json

clean:
	rm -rf vortex_afu.h $(BUILD_DIR)
