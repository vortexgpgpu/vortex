ASE_BUILD_DIR=build_ase
FPGA_BUILD_DIR=build_fpga
RTL_DIR=../../rtl

ifeq (, $(shell which qsub-synth))
	RUN_SYNTH=$(OPAE_PLATFORM_ROOT)/bin/run.sh > build.log 2>&1 &
else
	RUN_SYNTH=qsub-synth
endif

all: ase-1c

gen_sources_a10:	
	./gen_sources.sh arria10 > sources.txt

gen_sources_s10:	
	./gen_sources.sh stratix10 > sources.txt

ase-1c: gen_sources_a10 setup-ase-1c 
	make -C $(ASE_BUILD_DIR)_1c
	cp $(RTL_DIR)/fp_cores/altera/arria10/*.hex $(ASE_BUILD_DIR)_1c/work

ase-2c: gen_sources_a10 setup-ase-2c
	make -C $(ASE_BUILD_DIR)_2c
	cp $(RTL_DIR)/fp_cores/altera/arria10/*.hex $(ASE_BUILD_DIR)_2c/work

ase-4c: gen_sources_a10 setup-ase-4c
	make -C $(ASE_BUILD_DIR)_4c
	cp $(RTL_DIR)/fp_cores/altera/arria10/*.hex $(ASE_BUILD_DIR)_4c/work

setup-ase-1c: $(ASE_BUILD_DIR)_1c/Makefile

setup-ase-2c: $(ASE_BUILD_DIR)_2c/Makefile

setup-ase-4c: $(ASE_BUILD_DIR)_4c/Makefile

$(ASE_BUILD_DIR)_1c/Makefile:
	afu_sim_setup -s sources_1c.txt $(ASE_BUILD_DIR)_1c

$(ASE_BUILD_DIR)_2c/Makefile:
	afu_sim_setup -s sources_2c.txt $(ASE_BUILD_DIR)_2c

$(ASE_BUILD_DIR)_4c/Makefile:
	afu_sim_setup -s sources_4c.txt $(ASE_BUILD_DIR)_4c

fpga-1c: gen_sources_a10 setup-fpga-1c
	cd $(FPGA_BUILD_DIR)_1c && $(RUN_SYNTH)

fpga-2c: gen_sources_a10 setup-fpga-2c
	cd $(FPGA_BUILD_DIR)_2c && $(RUN_SYNTH)

fpga-4c: gen_sources_a10 setup-fpga-4c
	cd $(FPGA_BUILD_DIR)_4c && $(RUN_SYNTH)

fpga-8c: gen_sources_a10 setup-fpga-8c
	cd $(FPGA_BUILD_DIR)_8c && $(RUN_SYNTH)

fpga-16c: gen_sources_a10 setup-fpga-16c
	cd $(FPGA_BUILD_DIR)_16c && $(RUN_SYNTH)

fpga-32c: gen_sources_s10 setup-fpga-32c
	cd $(FPGA_BUILD_DIR)_32c && $(RUN_SYNTH)

fpga-64c: gen_sources_s10 setup-fpga-64c
	cd $(FPGA_BUILD_DIR)_64c && $(RUN_SYNTH)
	
setup-fpga-1c: $(FPGA_BUILD_DIR)_1c/build/dcp.qpf

setup-fpga-2c: $(FPGA_BUILD_DIR)_2c/build/dcp.qpf

setup-fpga-4c: $(FPGA_BUILD_DIR)_4c/build/dcp.qpf

setup-fpga-8c: $(FPGA_BUILD_DIR)_8c/build/dcp.qpf

setup-fpga-16c: $(FPGA_BUILD_DIR)_16c/build/dcp.qpf

setup-fpga-32c: $(FPGA_BUILD_DIR)_32c/build/dcp.qpf

setup-fpga-64c: $(FPGA_BUILD_DIR)_64c/build/dcp.qpf

$(FPGA_BUILD_DIR)_1c/build/dcp.qpf:
	afu_synth_setup -s sources_1c.txt $(FPGA_BUILD_DIR)_1c

$(FPGA_BUILD_DIR)_2c/build/dcp.qpf:
	afu_synth_setup -s sources_2c.txt $(FPGA_BUILD_DIR)_2c

$(FPGA_BUILD_DIR)_4c/build/dcp.qpf:
	afu_synth_setup -s sources_4c.txt $(FPGA_BUILD_DIR)_4c

$(FPGA_BUILD_DIR)_8c/build/dcp.qpf:
	afu_synth_setup -s sources_8c.txt $(FPGA_BUILD_DIR)_8c

$(FPGA_BUILD_DIR)_16c/build/dcp.qpf:
	afu_synth_setup -s sources_16c.txt $(FPGA_BUILD_DIR)_16c

$(FPGA_BUILD_DIR)_32c/build/dcp.qpf:
	afu_synth_setup -s sources_32c.txt $(FPGA_BUILD_DIR)_32c

$(FPGA_BUILD_DIR)_64c/build/dcp.qpf:
	afu_synth_setup -s sources_64c.txt $(FPGA_BUILD_DIR)_64c

run-ase-1c:
	cd $(ASE_BUILD_DIR)_1c && make sim

run-ase-2c:
	cd $(ASE_BUILD_DIR)_2c && make sim

run-ase-4c:
	cd $(ASE_BUILD_DIR)_4c && make sim

clean-ase-1c:
	rm -rf $(ASE_BUILD_DIR)_1c sources.txt

clean-ase-2c:
	rm -rf $(ASE_BUILD_DIR)_2c sources.txt

clean-ase-4c:
	rm -rf $(ASE_BUILD_DIR)_4c sources.txt

clean-fpga-1c:
	rm -rf $(FPGA_BUILD_DIR)_1c sources.txt
	
clean-fpga-2c:
	rm -rf $(FPGA_BUILD_DIR)_2c sources.txt

clean-fpga-4c:
	rm -rf $(FPGA_BUILD_DIR)_4c sources.txt

clean-fpga-8c:
	rm -rf $(FPGA_BUILD_DIR)_8c sources.txt

clean-fpga-16c:
	rm -rf $(FPGA_BUILD_DIR)_16c sources.txt

clean-fpga-32c:
	rm -rf $(FPGA_BUILD_DIR)_32c sources.txt

clean-fpga-64c:
	rm -rf $(FPGA_BUILD_DIR)_64c sources.txt

clean: clean-ase-1c	clean-ase-2c clean-ase-4c clean-fpga-1c clean-fpga-2c clean-fpga-4c clean-fpga-8c clean-fpga-16c clean-fpga-32c clean-fpga-64c
	rm sources.txt