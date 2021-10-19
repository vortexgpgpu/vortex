DEVICE_FAMILY ?= arria10
BUILD_DIR ?= build_$(DEVICE_FAMILY)
RTL_DIR=../../rtl

ifeq ($(shell which qsub-synth),)
	RUN_SYNTH=$(OPAE_PLATFORM_ROOT)/bin/run.sh > build.log 2>&1 &
else
	RUN_SYNTH=qsub-synth
endif

# control RTL debug tracing states
DBG_TRACE_FLAGS += -DDBG_TRACE_PIPELINE  
DBG_TRACE_FLAGS += -DDBG_TRACE_CORE_ICACHE
DBG_TRACE_FLAGS += -DDBG_TRACE_CORE_DCACHE
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_BANK 
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_MSHR
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_TAG
DBG_TRACE_FLAGS += -DDBG_TRACE_CACHE_DATA
DBG_TRACE_FLAGS += -DDBG_TRACE_MEM
DBG_TRACE_FLAGS += -DDBG_TRACE_OPAE
DBG_TRACE_FLAGS += -DDBG_TRACE_AVS
DBG_TRACE_FLAGS += -DDBG_TRACE_SCOPE
DBG_TRACE_FLAGS += -DDBG_TRACE_TEX

DBG_FLAGS += $(DBG_TRACE_FLAGS)
DBG_FLAGS += -DDBG_CACHE_REQ_INFO

CONFIG1  := -DNUM_CLUSTERS=1 -DNUM_CORES=1 -DL2_ENABLE=0 -DL3_ENABLE=0 $(CONFIGS)
CONFIG2  := -DNUM_CLUSTERS=1 -DNUM_CORES=2 -DL2_ENABLE=0 -DL3_ENABLE=0 $(CONFIGS)
CONFIG4  := -DNUM_CLUSTERS=1 -DNUM_CORES=4 -DL2_ENABLE=1 -DL3_ENABLE=0 -DICACHE_SIZE=8192 -DDCACHE_SIZE=8192 -DL2_CACHE_SIZE=131072 $(CONFIGS)
CONFIG8  := -DNUM_CLUSTERS=1 -DNUM_CORES=8 -DL2_ENABLE=1 -DL3_ENABLE=0 -DICACHE_SIZE=8192 -DDCACHE_SIZE=8192 -DL2_CACHE_SIZE=131072 $(CONFIGS)
CONFIG16 := -DNUM_CLUSTERS=4 -DNUM_CORES=4 -DL2_ENABLE=0 -DL3_ENABLE=1 -DICACHE_SIZE=8192 -DDCACHE_SIZE=8192 -DL3_CACHE_SIZE=262144 $(CONFIGS)
CONFIG32 := -DNUM_CLUSTERS=4 -DNUM_CORES=8 -DL2_ENABLE=0 -DL3_ENABLE=1 -DICACHE_SIZE=8192 -DDCACHE_SIZE=8192 -DL3_CACHE_SIZE=262144 $(CONFIGS)
CONFIG64 := -DNUM_CLUSTERS=8 -DNUM_CORES=8 -DL2_ENABLE=0 -DL3_ENABLE=1 -DICACHE_SIZE=8192 -DDCACHE_SIZE=8192 -DL3_CACHE_SIZE=524288 $(CONFIGS)

FPU_INCLUDE = -I$(RTL_DIR)/fp_cores -I$(RTL_DIR)/fp_cores/altera/$(DEVICE_FAMILY)
TEX_INCLUDE = -I$(RTL_DIR)/tex_unit
RTL_INCLUDE = -I$(RTL_DIR) -I$(RTL_DIR)/libs -I$(RTL_DIR)/interfaces -I$(RTL_DIR)/cache -I$(RTL_DIR)/afu $(FPU_INCLUDE) $(TEX_INCLUDE)

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

$(BUILD_DIR)_ase_1c/Makefile:
	afu_sim_setup -s setup.cfg $(BUILD_DIR)_ase_1c

$(BUILD_DIR)_ase_2c/Makefile:
	afu_sim_setup -s setup.cfg $(BUILD_DIR)_ase_2c

$(BUILD_DIR)_ase_4c/Makefile:
	afu_sim_setup -s setup.cfg $(BUILD_DIR)_ase_4c

$(BUILD_DIR)_fpga_1c/build/dcp.qpf:
	afu_synth_setup -s setup.cfg $(BUILD_DIR)_fpga_1c

$(BUILD_DIR)_fpga_2c/build/dcp.qpf:
	afu_synth_setup -s setup.cfg $(BUILD_DIR)_fpga_2c

$(BUILD_DIR)_fpga_4c/build/dcp.qpf:
	afu_synth_setup -s setup.cfg $(BUILD_DIR)_fpga_4c

$(BUILD_DIR)_fpga_8c/build/dcp.qpf:
	afu_synth_setup -s setup.cfg $(BUILD_DIR)_fpga_8c

$(BUILD_DIR)_fpga_16c/build/dcp.qpf:
	afu_synth_setup -s setup.cfg $(BUILD_DIR)_fpga_16c

$(BUILD_DIR)_fpga_32c/build/dcp.qpf:
	afu_synth_setup -s setup.cfg $(BUILD_DIR)_fpga_32c

$(BUILD_DIR)_fpga_64c/build/dcp.qpf:
	afu_synth_setup -s setup.cfg $(BUILD_DIR)_fpga_64c

gen-sources-1c:
	./gen_sources.sh $(CFLAGS) $(CONFIG1) > sources.txt

gen-sources-2c:
	./gen_sources.sh $(CFLAGS) $(CONFIG2) > sources.txt

gen-sources-4c:
	./gen_sources.sh $(CFLAGS) $(CONFIG4) > sources.txt

gen-sources-8c:
	./gen_sources.sh $(CFLAGS) $(CONFIG8) > sources.txt

gen-sources-16c:
	./gen_sources.sh $(CFLAGS) $(CONFIG16) > sources.txt

gen-sources-32c:
	./gen_sources.sh $(CFLAGS) $(CONFIG32) > sources.txt

gen-sources-64c:
	./gen_sources.sh $(CFLAGS) $(CONFIG64) > sources.txt

# setup

setup-ase-1c: $(BUILD_DIR)_ase_1c/Makefile

setup-ase-2c: $(BUILD_DIR)_ase_2c/Makefile

setup-ase-4c: $(BUILD_DIR)_ase_4c/Makefile

setup-fpga-1c: $(BUILD_DIR)_fpga_1c/build/dcp.qpf

setup-fpga-2c: $(BUILD_DIR)_fpga_2c/build/dcp.qpf

setup-fpga-4c: $(BUILD_DIR)_fpga_4c/build/dcp.qpf

setup-fpga-8c: $(BUILD_DIR)_fpga_8c/build/dcp.qpf

setup-fpga-16c: $(BUILD_DIR)_fpga_16c/build/dcp.qpf

setup-fpga-32c: $(BUILD_DIR)_fpga_32c/build/dcp.qpf

setup-fpga-64c: $(BUILD_DIR)_fpga_64c/build/dcp.qpf

# build

ase-1c: gen-sources-1c setup-ase-1c
	make -C $(BUILD_DIR)_ase_1c
	cp $(RTL_DIR)/fp_cores/altera/$(DEVICE_FAMILY)/*.hex $(BUILD_DIR)_ase_1c/work

ase-2c: gen-sources-2c setup-ase-2c
	make -C $(BUILD_DIR)_ase_2c
	cp $(RTL_DIR)/fp_cores/altera/$(DEVICE_FAMILY)/*.hex $(BUILD_DIR)_ase_2c/work

ase-4c: gen-sources-4c setup-ase-4c
	make -C $(BUILD_DIR)_ase_4c
	cp $(RTL_DIR)/fp_cores/altera/$(DEVICE_FAMILY)/*.hex $(BUILD_DIR)_ase_4c/work

fpga-1c: gen-sources-1c setup-fpga-1c
	cd $(BUILD_DIR)_fpga_1c && $(RUN_SYNTH)

fpga-2c: gen-sources-2c setup-fpga-2c
	cd $(BUILD_DIR)_fpga_2c && $(RUN_SYNTH)

fpga-4c: gen-sources-4c setup-fpga-4c
	cd $(BUILD_DIR)_fpga_4c && $(RUN_SYNTH)

fpga-8c: gen-sources-8c setup-fpga-8c
	cd $(BUILD_DIR)_fpga_8c && $(RUN_SYNTH)

fpga-16c: gen-sources-16c setup-fpga-16c
	cd $(BUILD_DIR)_fpga_16c && $(RUN_SYNTH)

fpga-32c: gen-sources-32c setup-fpga-32c
	cd $(BUILD_DIR)_fpga_32c && $(RUN_SYNTH)

fpga-64c: gen-sources-64c setup-fpga-64c
	cd $(BUILD_DIR)_fpga_64c && $(RUN_SYNTH)

# cleanup

clean-ase-1c:
	rm -rf $(BUILD_DIR)_ase_1c sources.txt

clean-ase-2c:
	rm -rf $(BUILD_DIR)_ase_2c sources.txt

clean-ase-4c:
	rm -rf $(BUILD_DIR)_ase_4c sources.txt

clean-fpga-1c:
	rm -rf $(BUILD_DIR)_fpga_1c sources.txt
	
clean-fpga-2c:
	rm -rf $(BUILD_DIR)_fpga_2c sources.txt

clean-fpga-4c:
	rm -rf $(BUILD_DIR)_fpga_4c sources.txt

clean-fpga-8c:
	rm -rf $(BUILD_DIR)_fpga_8c sources.txt

clean-fpga-16c:
	rm -rf $(BUILD_DIR)_fpga_16c sources.txt

clean-fpga-32c:
	rm -rf $(BUILD_DIR)_fpga_32c sources.txt

clean-fpga-64c:
	rm -rf $(BUILD_DIR)_fpga_64c sources.txt

clean: vortex_afu.h clean-ase-1c clean-ase-2c clean-ase-4c clean-fpga-1c clean-fpga-2c clean-fpga-4c clean-fpga-8c clean-fpga-16c clean-fpga-32c clean-fpga-64c