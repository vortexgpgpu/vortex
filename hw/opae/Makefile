ASE_BUILD_DIR=build_ase

FPGA_BUILD_DIR=build_fpga

all: ase-1c

sources.txt: 
	./gen_sources.sh

ase-1c: setup-ase-1c sources.txt
	make -C $(ASE_BUILD_DIR)_1c

ase-2c: setup-ase-2c sources.txt
	make -C $(ASE_BUILD_DIR)_2c

ase-4c: setup-ase-4c sources.txt
	make -C $(ASE_BUILD_DIR)_4c

setup-ase-1c: $(ASE_BUILD_DIR)_1c/Makefile

setup-ase-2c: $(ASE_BUILD_DIR)_2c/Makefile

setup-ase-4c: $(ASE_BUILD_DIR)_4c/Makefile

$(ASE_BUILD_DIR)_1c/Makefile:
	afu_sim_setup -s sources_1c.txt $(ASE_BUILD_DIR)_1c

$(ASE_BUILD_DIR)_2c/Makefile:
	afu_sim_setup -s sources_2c.txt $(ASE_BUILD_DIR)_2c

$(ASE_BUILD_DIR)_4c/Makefile:
	afu_sim_setup -s sources_4c.txt $(ASE_BUILD_DIR)_4c

fpga-1c: setup-fpga-1c sources.txt
	cd $(FPGA_BUILD_DIR)_1c && qsub-synth

fpga-2c: setup-fpga-2c sources.txt
	cd $(FPGA_BUILD_DIR)_2c && qsub-synth

fpga-4c: setup-fpga-4c sources.txt
	cd $(FPGA_BUILD_DIR)_4c && qsub-synth
	
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
	rm -rf $(ASE_BUILD_DIR)_1c

clean-ase-2c:
	rm -rf $(ASE_BUILD_DIR)_2c

clean-ase-4c:
	rm -rf $(ASE_BUILD_DIR)_4c

clean-fpga-1c:
	rm -rf $(FPGA_BUILD_DIR)_1c
	
clean-fpga-2c:
	rm -rf $(FPGA_BUILD_DIR)_2c

clean-fpga-4c:
	rm -rf $(FPGA_BUILD_DIR)_4c

