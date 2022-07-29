DEVICE_FAMILY ?= arria10
BUILD_DIR ?= build
RTL_DIR = ../../../rtl
DPI_DIR = ../../../dpi
AFU_DIR = ../../../afu/opae
IP_DIR  = ../ip/$(DEVICE_FAMILY)

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
RTL_INCLUDE = -I$(RTL_DIR) -I$(DPI_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/cache -I$(AFU_DIR) -I$(IP_DIR)
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

all: vortex_afu.h ase-1c

# AFU info from JSON file, including AFU UUID
vortex_afu.h: vortex_afu.json
	afu_json_mgr json-info --afu-json=$^ --c-hdr=$@

$(BUILD_DIR)_$(DEVICE_FAMILY)_ase_1c/Makefile:
	afu_sim_setup -s setup.cfg $(BUILD_DIR)_$(DEVICE_FAMILY)_ase_1c

$(BUILD_DIR)_$(DEVICE_FAMILY)_ase_2c/Makefile:
	afu_sim_setup -s setup.cfg $(BUILD_DIR)_$(DEVICE_FAMILY)_ase_2c

$(BUILD_DIR)_$(DEVICE_FAMILY)_ase_4c/Makefile:
	afu_sim_setup -s setup.cfg $(BUILD_DIR)_$(DEVICE_FAMILY)_ase_4c

$(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_1c/build/dcp.qpf:
	afu_synth_setup -s setup.cfg $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_1c

$(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_2c/build/dcp.qpf:
	afu_synth_setup -s setup.cfg $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_2c

$(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_4c/build/dcp.qpf:
	afu_synth_setup -s setup.cfg $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_4c

$(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_8c/build/dcp.qpf:
	afu_synth_setup -s setup.cfg $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_8c

$(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_16c/build/dcp.qpf:
	afu_synth_setup -s setup.cfg $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_16c

$(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_32c/build/dcp.qpf:
	afu_synth_setup -s setup.cfg $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_32c

$(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_64c/build/dcp.qpf:
	afu_synth_setup -s setup.cfg $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_64c

gen-sources-ase-1c:
	./gen_sources.sh $(CFLAGS) $(CONFIG1) -DSIMULATION > sources.txt

gen-sources-ase-2c:
	./gen_sources.sh $(CFLAGS) $(CONFIG2) -DSIMULATION > sources.txt

gen-sources-ase-4c:
	./gen_sources.sh $(CFLAGS) $(CONFIG4) -DSIMULATION > sources.txt

gen-sources-fpga-1c:
	./gen_sources.sh $(CFLAGS) $(CONFIG1) > sources.txt

gen-sources-fpga-2c:
	./gen_sources.sh $(CFLAGS) $(CONFIG2) > sources.txt

gen-sources-fpga-4c:
	./gen_sources.sh $(CFLAGS) $(CONFIG4) > sources.txt

gen-sources-fpga-8c:
	./gen_sources.sh $(CFLAGS) $(CONFIG8) > sources.txt

gen-sources-fpga-16c:
	./gen_sources.sh $(CFLAGS) $(CONFIG16) > sources.txt

gen-sources-fpga-32c:
	./gen_sources.sh $(CFLAGS) $(CONFIG32) > sources.txt

gen-sources-fpga-64c:
	./gen_sources.sh $(CFLAGS) $(CONFIG64) > sources.txt

# setup

setup-ase-1c: $(BUILD_DIR)_$(DEVICE_FAMILY)_ase_1c/Makefile

setup-ase-2c: $(BUILD_DIR)_$(DEVICE_FAMILY)_ase_2c/Makefile

setup-ase-4c: $(BUILD_DIR)_$(DEVICE_FAMILY)_ase_4c/Makefile

setup-fpga-1c: $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_1c/build/dcp.qpf

setup-fpga-2c: $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_2c/build/dcp.qpf

setup-fpga-4c: $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_4c/build/dcp.qpf

setup-fpga-8c: $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_8c/build/dcp.qpf

setup-fpga-16c: $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_16c/build/dcp.qpf

setup-fpga-32c: $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_32c/build/dcp.qpf

setup-fpga-64c: $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_64c/build/dcp.qpf

# build

ase-1c: gen-sources-ase-1c setup-ase-1c
	make -C $(BUILD_DIR)_$(DEVICE_FAMILY)_ase_1c
	cp $(RTL_DIR)/fpu_unit/altera/$(DEVICE_FAMILY)/*.hex $(BUILD_DIR)_$(DEVICE_FAMILY)_ase_1c/work

ase-2c: gen-sources-ase-2c setup-ase-2c
	make -C $(BUILD_DIR)_$(DEVICE_FAMILY)_ase_2c
	cp $(RTL_DIR)/fpu_unit/altera/$(DEVICE_FAMILY)/*.hex $(BUILD_DIR)_$(DEVICE_FAMILY)_ase_2c/work

ase-4c: gen-sources-ase-4c setup-ase-4c
	make -C $(BUILD_DIR)_$(DEVICE_FAMILY)_ase_4c
	cp $(RTL_DIR)/fpu_unit/altera/$(DEVICE_FAMILY)/*.hex $(BUILD_DIR)_$(DEVICE_FAMILY)_ase_4c/work

fpga-1c: gen-sources-fpga-1c setup-fpga-1c
	cd $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_1c && $(RUN_SYNTH)

fpga-2c: gen-sources-fpga-2c setup-fpga-2c
	cd $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_2c && $(RUN_SYNTH)

fpga-4c: gen-sources-fpga-4c setup-fpga-4c
	cd $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_4c && $(RUN_SYNTH)

fpga-8c: gen-sources-fpga-8c setup-fpga-8c
	cd $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_8c && $(RUN_SYNTH)

fpga-16c: gen-sources-fpga-16c setup-fpga-16c
	cd $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_16c && $(RUN_SYNTH)

fpga-32c: gen-sources-fpga-32c setup-fpga-32c
	cd $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_32c && $(RUN_SYNTH)

fpga-64c: gen-sources-fpga-64c setup-fpga-64c
	cd $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_64c && $(RUN_SYNTH)

# cleanup

clean-ase-1c:
	rm -rf $(BUILD_DIR)_$(DEVICE_FAMILY)_ase_1c sources.txt

clean-ase-2c:
	rm -rf $(BUILD_DIR)_$(DEVICE_FAMILY)_ase_2c sources.txt

clean-ase-4c:
	rm -rf $(BUILD_DIR)_$(DEVICE_FAMILY)_ase_4c sources.txt

clean-fpga-1c:
	rm -rf $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_1c sources.txt
	
clean-fpga-2c:
	rm -rf $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_2c sources.txt

clean-fpga-4c:
	rm -rf $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_4c sources.txt

clean-fpga-8c:
	rm -rf $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_8c sources.txt

clean-fpga-16c:
	rm -rf $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_16c sources.txt

clean-fpga-32c:
	rm -rf $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_32c sources.txt

clean-fpga-64c:
	rm -rf $(BUILD_DIR)_$(DEVICE_FAMILY)_fpga_64c sources.txt

clean: vortex_afu.h clean-ase-1c clean-ase-2c clean-ase-4c clean-fpga-1c clean-fpga-2c clean-fpga-4c clean-fpga-8c clean-fpga-16c clean-fpga-32c clean-fpga-64c