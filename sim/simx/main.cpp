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
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include "processor.h"
#include "mem.h"
#include "constants.h"
#include <util.h>
#include <simobject.h>
#include "core.h"
#include "scheduler.h"
#include "VX_types.h"

using namespace vortex;

static void show_usage() {
   std::cout << "Usage: [-s: stats] [-h: help] <program>" << std::endl;
}

bool showStats = false;
const char* program = nullptr;

static void parse_args(int argc, char **argv) {
  	int c;
  	while ((c = getopt(argc, argv, "sh")) != -1) {
    	switch (c) {
      case 's':
        showStats = true;
        break;
    	case 'h':
      	show_usage();
      	exit(0);
    		break;
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

  {
    // create memory module
    RAM ram(0, MEM_PAGE_SIZE);

    // create processor
    Processor processor;

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
    if (program) {
      std::string program_ext(fileExtension(program));
      if (program_ext == "vxbin") {
        ram.loadVxImage(program);
      } else if (program_ext == "bin") {
        ram.loadBinImage(program, startup_addr);
      } else if (program_ext == "hex") {
        ram.loadHexImage(program);
      } else {
        std::cerr << "Error: only *.vxbin, *.bin or *.hex images supported." << std::endl;
        return -1;
      }
    }
  #ifndef NDEBUG
    std::cout << "[VXDRV] START: program=" << program << std::endl;
  #endif
    // run simulation
    processor.run();

    if (debug_mode) {
      // Debug mode: run RBB server in a tick loop until OpenOCD disconnects
      // and the program has completed naturally.
      std::cout << "[DEBUG] Starting debug mode on port " << rbb_port << std::endl;

      DebugModule::set_verbose_logging(debug_verbose);

      // Reset processor and kick the KMU. In normal run() mode these are
      // both done at the top of run(); debug mode replaces run() with its
      // own tick loop, so it must do them explicitly.
      processor.reset();
      processor.start_kmu();

      Core* core = processor.get_first_core();
      if (core == nullptr) {
        std::cerr << "[DEBUG] no cores configured" << std::endl;
        return -1;
      }

    // read exitcode from @MPM.1
    ram.read(&exitcode, IO_EXIT_CODE, 4);
  }

  return exitcode;
}
