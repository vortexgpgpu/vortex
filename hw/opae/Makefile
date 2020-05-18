
ASE_BUILD_DIR=build_ase
FPGA_BUILD_DIR=build_fpga

all: ase fpga

ase: setup-ase
	make -C $(ASE_BUILD_DIR)

fpga: setup-fpga
	cd $(FPGA_BUILD_DIR) && qsub-synth

setup-ase: $(ASE_BUILD_DIR)/Makefile

setup-fpga: $(FPGA_BUILD_DIR)/build/dcp.qpf

$(ASE_BUILD_DIR)/Makefile:
	afu_sim_setup -s sources.txt $(ASE_BUILD_DIR)

$(FPGA_BUILD_DIR)/build/dcp.qpf:
	afu_synth_setup -s sources.txt $(FPGA_BUILD_DIR)

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