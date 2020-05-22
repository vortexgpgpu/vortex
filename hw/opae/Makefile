
ASE_BUILD_DIR=build_ase
FPGA_BUILD_DIR=build_fpga

all: ase fpga fpga-1c

ase: setup-ase
	make -C $(ASE_BUILD_DIR)

setup-ase: $(ASE_BUILD_DIR)/Makefile

$(ASE_BUILD_DIR)/Makefile:
	afu_sim_setup -s sources.txt $(ASE_BUILD_DIR)

fpga: setup-fpga
	cd $(FPGA_BUILD_DIR) && qsub-synth

setup-fpga: $(FPGA_BUILD_DIR)/build/dcp.qpf

$(FPGA_BUILD_DIR)/build/dcp.qpf:
	afu_synth_setup -s sources.txt $(FPGA_BUILD_DIR)

fpga-1c: setup-fpga-1c
	cd $(FPGA_BUILD_DIR)_1c && qsub-synth
	
setup-fpga-1c: $(FPGA_BUILD_DIR)_1c/build/dcp.qpf

$(FPGA_BUILD_DIR)_1c/build/dcp.qpf:
	afu_synth_setup -s sources_1c.txt $(FPGA_BUILD_DIR)_1c

run-ase:
	cd $(ASE_BUILD_DIR) && make sim

wave:
	vsim -view $(ASE_BUILD_DIR)/work/vsim.wlf -do wave.do

run-fpga:
	# TODO

clean-ase:
	rm -rf $(ASE_BUILD_DIR)

clean-fpga:
	rm -rf $(FPGA_BUILD_DIR)