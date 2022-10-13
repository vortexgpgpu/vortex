DEVICE_FAMILY ?= arria10
PREFIX ?= build
NUM_CORES ?= 1
RTL_DIR = ../../../rtl
DPI_DIR = ../../../dpi
AFU_DIR = ../../../afu/opae
IP_DIR  = ../../../ip/altera

IP_CACHE_DIR = ../ip_cache/$(DEVICE_FAMILY)

BUILD_DIR = $(PREFIX)_$(DEVICE_FAMILY)

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

CONFIGS += -DEXT_GFX_ENABLE

#CONFIGS += -DNUM_ROP_UNITS=2
#CONFIGS += -DNUM_TEX_UNITS=4
#CONFIGS += -DNUM_RASTER_UNITS=2
#CONFIGS += -DTCACHE_NUM_BANKS=1
#CONFIGS += -DOCACHE_NUM_BANKS=1
#CONFIGS += -DOCACHE_NUM_BANKS=1

#CONFIGS += -DL1_DISABLE
#CONFIGS += -DSM_DISABLE
#CONFIGS += -DRCACHE_DISABLE
#CONFIGS += -DOCACHE_DISABLE
#CONFIGS += -DTCACHE_DISABLE

ifeq ($(DEVICE_FAMILY), stratix10)
    CONFIGS += -DALTERA_S10
endif
ifeq ($(DEVICE_FAMILY), arria10)
    CONFIGS += -DALTERA_A10
endif

CONFIG1  := -DNUM_CLUSTERS=1 -DNUM_CORES=1 $(CONFIGS)
CONFIG2  := -DNUM_CLUSTERS=1 -DNUM_CORES=2 $(CONFIGS)
CONFIG4  := -DNUM_CLUSTERS=1 -DNUM_CORES=4  -DL2_ENABLE $(CONFIGS)
CONFIG8  := -DNUM_CLUSTERS=1 -DNUM_CORES=8  -DL2_ENABLE $(CONFIGS)
CONFIG16 := -DNUM_CLUSTERS=1 -DNUM_CORES=16 -DL2_ENABLE $(CONFIGS)
CONFIG32 := -DNUM_CLUSTERS=2 -DNUM_CORES=16 -DL2_ENABLE $(CONFIGS)
CONFIG64 := -DNUM_CLUSTERS=4 -DNUM_CORES=16 -DL2_ENABLE $(CONFIGS)

FPU_INCLUDE = -I$(RTL_DIR)/fpu_unit
TEX_INCLUDE = -I$(RTL_DIR)/tex_unit
RASTER_INCLUDE = -I$(RTL_DIR)/raster_unit
ROP_INCLUDE = -I$(RTL_DIR)/rop_unit
RTL_INCLUDE = -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/cache -I$(AFU_DIR) -I$(IP_DIR) -I$(IP_CACHE_DIR)
RTL_INCLUDE += $(FPU_INCLUDE) $(TEX_INCLUDE) $(RASTER_INCLUDE) $(ROP_INCLUDE)

CFLAGS += $(RTL_INCLUDE)

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

all: swconfig ip_gen

ip_gen: $(IP_CACHE_DIR)/ip_gen.log
$(IP_CACHE_DIR)/ip_gen.log:
	../ip_gen.sh

# AFU info from JSON file, including AFU UUID
swconfig: vortex_afu.h
vortex_afu.h: vortex_afu.json
	afu_json_mgr json-info --afu-json=$^ --c-hdr=$@

gen-sources-ase:
	./gen_sources.sh $(CFLAGS) $(CONFIG$(NUM_CORES)) -DSIMULATION > sources.txt

gen-sources-fpga:
	./gen_sources.sh $(CFLAGS) $(CONFIG$(NUM_CORES)) > sources.txt

setup-ase: $(BUILD_DIR)_ase_$(NUM_CORES)c/Makefile
$(BUILD_DIR)_ase_$(NUM_CORES)c/Makefile:
	afu_sim_setup -s setup.cfg $(BUILD_DIR)_ase_$(NUM_CORES)c

setup-fpga: $(BUILD_DIR)_fpga_$(NUM_CORES)c/build/dcp.qpf
$(BUILD_DIR)_fpga_$(NUM_CORES)c/build/dcp.qpf:
	afu_synth_setup -s setup.cfg $(BUILD_DIR)_fpga_$(NUM_CORES)c

# build

build-ase: ip_gen gen-sources-ase setup-ase	
	make -C $(BUILD_DIR)_ase_$(NUM_CORES)c
	cp $(IP_CACHE_DIR)/*.hex $(BUILD_DIR)_ase_$(NUM_CORES)c/work	

build-fpga: ip_gen gen-sources-fpga setup-fpga
	cd $(BUILD_DIR)_fpga_$(NUM_CORES)c && $(RUN_SYNTH)

# cleanup

clean-ase:  
	rm -rf vortex_afu.h $(BUILD_DIR)_ase_$(NUM_CORES)c

clean-fpga: 
	rm -rf vortex_afu.h $(BUILD_DIR)_fpga_$(NUM_CORES)c