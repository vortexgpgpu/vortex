
BUILD_DIR=build_ase

all: ase fpga

ase: setup-ase
	make -C $(BUILD_DIR)

fpga: setup-fpga
	cd build_fpga && qsub-synth

setup-ase: build_ase/Makefile

setup-fpga: build_fpga/build/dcp.qpf

build_ase/Makefile:
	afu_sim_setup --s sources.txt build_ase

build_fpga/build/dcp.qpf:
	afu_synth_setup -s sources.txt build_fpga

run-ase:
	cd build_ase && make sim

wave:
	vsim -view build_ase/work/vsim.wlf -do wave.do

run-fpga:
	# TODO

clean-ase:
	rm -rf build_ase

clean-fpga:
	rm -rf build_fpga