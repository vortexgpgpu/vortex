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
#include "dtm/debug_module.h"
#include "dtm/jtag_dtm.h"
#include "dtm/remote_bitbang.h"

using namespace vortex;

static void show_usage() {
   std::cout << "Usage: [-s: stats] [-d: debug-mode] [-p <port>: RBB port] [-V: verbose debug logging] [-h: help] <program>" << std::endl;
}

bool showStats = false;
bool debug_mode = false;
bool debug_verbose = false;  // Verbose debug module logging
uint16_t rbb_port = 9823;    // Default OpenOCD remote bitbang port
const char* program = nullptr;

static void parse_args(int argc, char **argv) {
  	int c;
  	while ((c = getopt(argc, argv, "shdp:V")) != -1) {
    	switch (c) {
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
    {
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

      // Prime warp 0's PC to the configured startup address. Unlike the
      // legacy Emulator (which exposed startup_addr through reset()), the
      // v3 Scheduler only sets warp PC when the KMU dispatches a CTA.
      // For DTM debug we want PC visible *before* any cycle has run, so
      // we initialize it directly. The DCR-driven dispatch path still
      // works on resume (KMU will re-dispatch through normal scheduling).
      core->dtm_set_pc(0, static_cast<vortex::Word>(startup_addr));

      DebugModule dm(core, &ram);

      // Halt the hart at startup so the debugger can control execution.
      // Set DPC to the initial PC so the first RR of DPC reads correctly.
      vortex::Word initial_pc = core->dtm_get_pc(0);
      dm.set_debug_mode_enabled(true);
      dm.direct_write_register(0x7B1, initial_pc);
      dm.halt_hart(0);

      jtag_dtm_t dtm(&dm);
      remote_bitbang_t rbb(rbb_port, &dtm);

      std::cout << "[DEBUG] Remote bitbang server ready. Waiting for OpenOCD connection..." << std::endl;

      bool program_completed_notified = false;
      bool ever_ran = false;
      while (true) {
        // Only advance the simulator while the hart is not halted by DTM.
        // (v3 has no Emulator gate; honoring DM's halt state at the tick
        // boundary is what implements halt/resume in this branch.)
        if (!dm.hart_is_halted()) {
          SimPlatform::instance().tick();
          if (processor.any_running()) {
            ever_ran = true;
          }
        }
        rbb.tick();

        // Once the program has actually started, detect natural completion
        // (no clusters running and no in-flight channel packets) and notify
        // DebugModule once. The debugger can still resume / inspect after.
        if (ever_ran && !program_completed_notified && !processor.any_running()) {
          dm.notify_program_completed(core->dtm_get_pc(0));
          program_completed_notified = true;
        }
      }
      // Unreachable in practice; debug mode is a long-running session.
    } else {
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
    }
  }

  return exitcode;
}
