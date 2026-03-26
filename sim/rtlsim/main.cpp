// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <fstream>
#include <iomanip>
#include <unistd.h>
#include <util.h>
#include <mem.h>
#include <VX_config.h>
#include <VX_types.h>
#include "processor.h"

#define RAM_PAGE_SIZE 4096

using namespace vortex;

static void show_usage() {
   std::cout << "Usage: [-h: help] <program>" << std::endl;
}

const char* program = nullptr;

static void parse_args(int argc, char **argv) {
  	int c;
  	while ((c = getopt(argc, argv, "rh")) != -1) {
    	switch (c) {
    	case 'h':
      	show_usage();
      	exit(0);
    	default:
      		show_usage();
      		exit(-1);
    	}
	}

	if (optind < argc) {
		program = argv[optind];
		std::cout << "Running " << program << "..." << std::endl;
	} else {
		show_usage();
      	exit(-1);
	}
}

int main(int argc, char **argv) {
	int exitcode = 0;

	parse_args(argc, argv);

	// create memory module
	vortex::RAM ram(0, RAM_PAGE_SIZE);

	// create processor
	vortex::Processor processor;

	// attach memory module
	processor.attach_ram(&ram);

	// setup base DCRs
	const uint64_t startup_addr(STARTUP_ADDR);
	processor.dcr_write(VX_DCR_KMU_STARTUP_ADDR0, startup_addr & 0xffffffff);
#if (XLEN == 64)
    processor.dcr_write(VX_DCR_KMU_STARTUP_ADDR1, startup_addr >> 32);
#endif
    processor.dcr_write(VX_DCR_KMU_STARTUP_ARG0, 0);
    processor.dcr_write(VX_DCR_KMU_STARTUP_ARG1, 0);
    processor.dcr_write(VX_DCR_KMU_GRID_DIM_X,   1);
    processor.dcr_write(VX_DCR_KMU_GRID_DIM_Y,   1);
    processor.dcr_write(VX_DCR_KMU_GRID_DIM_Z,   1);
    processor.dcr_write(VX_DCR_KMU_BLOCK_DIM_X,  1);
    processor.dcr_write(VX_DCR_KMU_BLOCK_DIM_Y,  1);
    processor.dcr_write(VX_DCR_KMU_BLOCK_DIM_Z,  1);
    processor.dcr_write(VX_DCR_KMU_LMEM_SIZE,    0);
    processor.dcr_write(VX_DCR_KMU_BLOCK_SIZE,   1);
    processor.dcr_write(VX_DCR_KMU_WARP_STEP_X,  NUM_THREADS);
    processor.dcr_write(VX_DCR_KMU_WARP_STEP_Y,  0);
    processor.dcr_write(VX_DCR_KMU_WARP_STEP_Z,  0);

	// load program
	{
		std::string program_ext(fileExtension(program));
		if (program_ext == "bin") {
			ram.loadBinImage(program, startup_addr);
		} else if (program_ext == "hex") {
			ram.loadHexImage(program);
		} else {
			std::cerr << "Error: only *.bin or *.hex images supported." << std::endl;
			return -1;
		}
	}
#ifndef NDEBUG
	std::cout << "[VXDRV] START: program=" << program << std::endl;
#endif
	// run simulation
	processor.run();

	// flush GPU caches before reading back results
	{
		uint32_t dummy;
		for (uint32_t cid = 0; cid < NUM_CORES * NUM_CLUSTERS; ++cid) {
			processor.dcr_read(VX_DCR_BASE_CACHE_FLUSH, cid, &dummy);
		}
	}

	// read exitcode from @MPM.1
  ram.read(&exitcode, IO_EXIT_CODE, 4);

	// Use _exit() to bypass destructors — Verilator's VerilatedScope destructor
	// calls scopeErase which crashes on strcmp with certain 64-bit module hierarchies.
	// The simulation is complete at this point; OS will reclaim all resources.
	_exit(exitcode);
}
