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
#include "core.h"
#include "VX_types.h"
#include "emulator.h"
#include "dtm/debug_module.h"
#include "dtm/jtag_dtm.h"
#include "dtm/remote_bitbang.h"

using namespace vortex;

static void show_usage() {
   std::cout << "Usage: [-c <cores>] [-w <warps>] [-t <threads>] [-v: vector-test] [-s: stats] [-d: debug-mode] [-p <port>: RBB port] [-V: verbose debug logging] [-h: help] <program>" << std::endl;
}

uint32_t num_threads = NUM_THREADS;
uint32_t num_warps = NUM_WARPS;
uint32_t num_cores = NUM_CORES;
bool showStats = false;
bool vector_test = false;
bool debug_mode = false;
bool debug_verbose = false;  // Verbose debug module logging
uint16_t rbb_port = 9823;  // Default OpenOCD remote bitbang port
const char* program = nullptr;

static void parse_args(int argc, char **argv) {
  	int c;
  	while ((c = getopt(argc, argv, "t:w:c:vshdp:V")) != -1) {
    	switch (c) {
      case 't':
        num_threads = atoi(optarg);
        break;
      case 'w':
        num_warps = atoi(optarg);
        break;
		  case 'c':
        num_cores = atoi(optarg);
        break;
      case 'v':
        vector_test = true;
        break;
      case 's':
        showStats = true;
        break;
      case 'd':
        debug_mode = true;
        break;
      case 'p':
        rbb_port = static_cast<uint16_t>(atoi(optarg));
        break;
      case 'V':
        debug_verbose = true;
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
    if (!debug_mode) {
      std::cout << "Running " << program << "..." << std::endl;
    }
	} else if (!debug_mode) {
		show_usage();
    exit(-1);
	}
}

int main(int argc, char **argv) {
  int exitcode = 0;

  parse_args(argc, argv);

  {
    // create processor configuation
    Arch arch(num_threads, num_warps, num_cores);

    // create memory module
    RAM ram(0, MEM_PAGE_SIZE);

    // create processor
    Processor processor(arch);

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

    if (debug_mode) {
      // Debug mode: run RBB server in infinite loop
      std::cout << "[DEBUG] Starting debug mode on port " << rbb_port << std::endl;
      
      // Set verbose logging for debug module based on command-line flag
      DebugModule::set_verbose_logging(debug_verbose);
      
      // Get emulator from processor
      Emulator* emulator = processor.get_first_emulator();
      
      // Reset emulator to read startup address from DCRs and initialize PC
      if (emulator != nullptr) {
        std::cout << "[DEBUG] Resetting emulator to initialize PC from DCRs..." << std::endl;
        emulator->reset();
        auto& warp0 = emulator->get_warp(0);
        std::cout << "[DEBUG] Emulator reset complete. PC = 0x" << std::hex << warp0.PC << std::dec << std::endl;
      }
      
      // Create debug module with emulator reference
      DebugModule dm(emulator);
      
      // Set debug module in emulator so it can check flags
      if (emulator != nullptr) {
        emulator->set_debug_module(&dm);
      }
      
      // Halt the program at startup so debugger can control execution
      // This ensures the program doesn't run until the debugger explicitly resumes it
      dm.set_debug_mode_enabled(true);
      if (emulator != nullptr) {
        // Update DPC with initial PC value before halting
        auto& warp0 = emulator->get_warp(0);
        vortex::Word initial_pc = warp0.PC;  // PC is already Word type
        dm.direct_write_register(0x7B1, initial_pc);  // Set DPC to initial PC
        // Note: We don't need to suspend the warp here because halt_requested_
        // check at the start of step() will prevent execution
      }
      // Halt the hart (cause 0 = reserved, but we use it for initial halt)
      // This sets halt_requested and is_halted flags, and updates DCSR
      dm.halt_hart(0);  // Cause 0 for initial halt state
      
      // Initialize and reset simulation platform
      SimPlatform::instance().initialize();
      SimPlatform::instance().reset();
      
      // Create JTAG DTM
      jtag_dtm_t dtm(&dm);
      
      // Create remote bitbang server
      remote_bitbang_t rbb(rbb_port, &dtm);
      
      std::cout << "[DEBUG] Remote bitbang server ready. Waiting for OpenOCD connection..." << std::endl;
      
      // Debug loop: advance simulation and handle JTAG communication
      while (true) {
        // Advance simulation by one cycle
        SimPlatform::instance().tick();
        // Handle JTAG/debugger communication
        rbb.tick();
      }
    } else {
      // run simulation
    #ifdef EXT_V_ENABLE
      // vector test exitcode is a special case
      if (vector_test) return (processor.run() != 1);
    #endif
      // else continue as normal
      processor.run();

    // flush GPU caches before reading back results
    {
      uint32_t dummy;
      for (uint32_t cid = 0; cid < num_cores * NUM_CLUSTERS; ++cid) {
        processor.dcr_read(VX_DCR_BASE_CACHE_FLUSH, cid, &dummy);
      }
    }

      // read exitcode from @MPM.1
    ram.read(&exitcode, IO_EXIT_CODE, 4);
    }
  }

  return exitcode;
}
