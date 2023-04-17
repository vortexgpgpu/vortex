DEVICE_FAMILY ?= arria10
PREFIX ?= build
TARGET ?= fpga
NUM_CORES ?= 1
SCRIPT_DIR = ../../../scripts
RTL_DIR = ../../../rtl
DPI_DIR = ../../../dpi
AFU_DIR = $(RTL_DIR)/afu/opae

IP_CACHE_DIR = ../ip_cache/$(DEVICE_FAMILY)

BUILD_DIR = $(PREFIX)_$(DEVICE_FAMILY)_$(TARGET)_$(NUM_CORES)c

ifeq ($(shell which qsub-synth),)
	RUN_SYNTH=$(OPAE_PLATFORM_ROOT)/bin/run.sh > build.log 2>&1 &
else
	RUN_SYNTH=qsub-synth
endif

# control RTL debug tracing states
DBG_TRACE_FLAGS += -DDBG_TRACE_CORE_PIPELINE  
DBG_TRACE_FLAGS += -DDBG_TRACE_CORE_ICACHE
DBG_TRACE_FLAGS += -DDBG_TRACE_CORE_DCACHE
DBG_TRACE_FLAGS += -DDBG_TRACE_CORE_MEM
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_BANK 
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_MSHR
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_TAG
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_DATA
DBG_TRACE_FLAGS += -DDBG_TRACE_AFU
DBG_TRACE_FLAGS += -DDBG_TRACE_TEX
DBG_TRACE_FLAGS += -DDBG_TRACE_RASTER
DBG_TRACE_FLAGS += -DDBG_TRACE_ROP

DBG_FLAGS += $(DBG_TRACE_FLAGS)

ifeq ($(DEVICE_FAMILY), stratix10)
    CONFIGS += -DALTERA_S10
endif
ifeq ($(DEVICE_FAMILY), arria10)
    CONFIGS += -DALTERA_A10
endif

# cluster configuration
CONFIGS_1c  := -DNUM_CLUSTERS=1 -DNUM_CORES=1
CONFIGS_2c  := -DNUM_CLUSTERS=1 -DNUM_CORES=2
CONFIGS_4c  := -DNUM_CLUSTERS=1 -DNUM_CORES=4  -DL2_ENABLE
CONFIGS_8c	:= -DNUM_CLUSTERS=1 -DNUM_CORES=8  -DL2_ENABLE
CONFIGS_16c	:= -DNUM_CLUSTERS=1 -DNUM_CORES=16 -DL2_ENABLE
CONFIGS_32c := -DNUM_CLUSTERS=2 -DNUM_CORES=16 -DL2_ENABLE
CONFIGS_64c := -DNUM_CLUSTERS=4 -DNUM_CORES=16 -DL2_ENABLE
CONFIGS += $(CONFIGS_$(NUM_CORES)c)

# include paths
FPU_INCLUDE = -I$(RTL_DIR)/fpu_unit
TEX_INCLUDE = -I$(RTL_DIR)/tex_unit
RASTER_INCLUDE = -I$(RTL_DIR)/raster_unit
ROP_INCLUDE = -I$(RTL_DIR)/rop_unit
RTL_INCLUDE = -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/core -I$(RTL_DIR)/cache -I$(AFU_DIR) -I$(IP_CACHE_DIR)
RTL_INCLUDE += $(FPU_INCLUDE) $(TEX_INCLUDE) $(RASTER_INCLUDE) $(ROP_INCLUDE)

# compilation flags
CFLAGS += -DSYNTHESIS -DQUARTUS
CFLAGS += $(CONFIGS)
CFLAGS += $(RTL_INCLUDE)

ifeq ($(TARGET), ase)
	CFLAGS += -DSIMULATION
endif

# Debugigng
ifdef DEBUG
	CFLAGS += $(DBG_FLAGS)
else    
	CFLAGS += -DNDEBUG
endif

# Enable scope analyzer
ifdef SCOPE
	CFLAGS += -DSCOPE
endif

# Enable perf counters
ifdef PERF
	CFLAGS += -DPERF_ENABLE
endif

all: swconfig ip-gen setup build

ip-gen: $(IP_CACHE_DIR)/ip-gen.log
$(IP_CACHE_DIR)/ip-gen.log:
	../ip_gen.sh $(IP_CACHE_DIR)

swconfig: vortex_afu.h
vortex_afu.h: vortex_afu.json
	afu_json_mgr json-info --afu-json=$^ --c-hdr=$@

setup:	
ifeq ($(TARGET), ase)
	afu_sim_setup -s setup.cfg $(BUILD_DIR)
else
	afu_synth_setup -s setup.cfg $(BUILD_DIR)
endif
	$(SCRIPT_DIR)/gen_sources.sh $(CFLAGS) -F$(BUILD_DIR)/src -Osources.txt

build: ip-gen setup
ifeq ($(TARGET), ase)
	make -C $(BUILD_DIR) > $(BUILD_DIR)/build.log 2>&1 &
else
	cd $(BUILD_DIR) && $(RUN_SYNTH)
endif

clean:  
	rm -rf vortex_afu.h $(BUILD_DIR)
