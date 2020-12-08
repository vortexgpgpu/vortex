ASE_BUILD_DIR=build_ase
FPGA_BUILD_DIR=build_fpga
RTL_DIR=../rtl

all: ase-1c

sources.txt:	
	./gen_sources.sh > sources.txt

gen_sources: sources.txt

ase-1c: gen_sources setup-ase-1c 
	make -C $(ASE_BUILD_DIR)_1c
	cp $(RTL_DIR)/fp_cores/altera/*.hex $(ASE_BUILD_DIR)_1c/work

ase-2c: gen_sources setup-ase-2c
	make -C $(ASE_BUILD_DIR)_2c
	cp $(RTL_DIR)/fp_cores/altera/*.hex $(ASE_BUILD_DIR)_2c/work

ase-4c: gen_sources setup-ase-4c
	make -C $(ASE_BUILD_DIR)_4c
	cp $(RTL_DIR)/fp_cores/altera/*.hex $(ASE_BUILD_DIR)_4c/work

setup-ase-1c: $(ASE_BUILD_DIR)_1c/Makefile

setup-ase-2c: $(ASE_BUILD_DIR)_2c/Makefile

setup-ase-4c: $(ASE_BUILD_DIR)_4c/Makefile

$(ASE_BUILD_DIR)_1c/Makefile: sources.txt
	afu_sim_setup -s sources_1c.txt $(ASE_BUILD_DIR)_1c

$(ASE_BUILD_DIR)_2c/Makefile: sources.txt
	afu_sim_setup -s sources_2c.txt $(ASE_BUILD_DIR)_2c

$(ASE_BUILD_DIR)_4c/Makefile: sources.txt
	afu_sim_setup -s sources_4c.txt $(ASE_BUILD_DIR)_4c

fpga-1c: gen_sources setup-fpga-1c
	cd $(FPGA_BUILD_DIR)_1c && qsub-synth
	cp $(RTL_DIR)/fp_cores/altera/*.hex $(FPGA_BUILD_DIR)_1c

fpga-2c: gen_sources setup-fpga-2c
	cd $(FPGA_BUILD_DIR)_2c && qsub-synth
	cp $(RTL_DIR)/fp_cores/altera/*.hex $(FPGA_BUILD_DIR)_2c

fpga-4c: gen_sources setup-fpga-4c
	cd $(FPGA_BUILD_DIR)_4c && qsub-synth
	cp $(RTL_DIR)/fp_cores/altera/*.hex $(FPGA_BUILD_DIR)_4c
	
setup-fpga-1c: $(FPGA_BUILD_DIR)_1c/build/dcp.qpf

setup-fpga-2c: $(FPGA_BUILD_DIR)_2c/build/dcp.qpf

setup-fpga-4c: $(FPGA_BUILD_DIR)_4c/build/dcp.qpf

$(FPGA_BUILD_DIR)_1c/build/dcp.qpf:
	afu_synth_setup -s sources_1c.txt $(FPGA_BUILD_DIR)_1c

$(FPGA_BUILD_DIR)_2c/build/dcp.qpf:
	afu_synth_setup -s sources_2c.txt $(FPGA_BUILD_DIR)_2c

$(FPGA_BUILD_DIR)_4c/build/dcp.qpf:
	afu_synth_setup -s sources_4c.txt $(FPGA_BUILD_DIR)_4c

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

clean: clean-ase-1c	clean-ase-2c clean-ase-4c clean-fpga-1c clean-fpga-2c clean-fpga-4c
	rm sources.txt