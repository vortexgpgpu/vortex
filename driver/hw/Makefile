
BUILD_DIR=build_sim

all: ase fpga

ase: build-setup
	make -C $(BUILD_DIR)

fpga: build-setup
	# TODO

build-setup: $(BUILD_DIR)/Makefile

$(BUILD_DIR)/Makefile:
	afu_sim_setup --sources=sources.txt --platform discrete_pcie3 $(BUILD_DIR) -f

run-ase:
	cd $(BUILD_DIR) && MENT_VSIM_OPT="-dpicpppath /usr/bin/gcc" make sim

run-fpga:
	# TODO

clean:
	rm -rf $(BUILD_DIR)