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

using namespace vortex;

static void show_usage() {
   std::cout << "Usage: [-c <cores>] [-w <warps>] [-t <threads>] [-r: riscv-test] [-s: stats] [-h: help] <program>" << std::endl;
}

uint32_t num_cores = NUM_CORES * NUM_CLUSTERS;
uint32_t num_warps = NUM_WARPS;
uint32_t num_threads = NUM_THREADS;
bool showStats = false;;
bool riscv_test = false;
const char* program = nullptr;

static void parse_args(int argc, char **argv) {
  	int c;
  	while ((c = getopt(argc, argv, "c:w:t:rsh?")) != -1) {
    	switch (c) {
		  case 'c':
        num_cores = atoi(optarg);
        break;
      case 'w':
        num_warps = atoi(optarg);
        break;
      case 't':
        num_threads = atoi(optarg);
        break;
      case 'r':
        riscv_test = true;
        break;
      case 's':
        showStats = true;
        break;
    	case 'h':
    	case '?':
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
    // create processor configuation
    Arch arch(num_cores, num_warps, num_threads);

    // create memory module
    RAM ram(RAM_PAGE_SIZE);

    // create processor
    Processor processor(arch);
  
    // attach memory module
    processor.attach_ram(&ram); 

	  // setup base DCRs
    processor.write_dcr(DCR_BASE_STARTUP_ADDR0, STARTUP_ADDR & 0xffffffff);
  #if (XLEN == 64)
    processor.write_dcr(DCR_BASE_STARTUP_ADDR1, STARTUP_ADDR >> 32);
  #endif
	  processor.write_dcr(DCR_BASE_MPM_CLASS, 0);

    // load program
    {      
      std::string program_ext(fileExtension(program));
      if (program_ext == "bin") {
        ram.loadBinImage(program, STARTUP_ADDR);
      } else if (program_ext == "hex") {
        ram.loadHexImage(program);
      } else {
        std::cout << "*** error: only *.bin or *.hex images supported." << std::endl;
        return -1;
      }
    }

    // run simulation
    exitcode = processor.run();
  } 

  if (riscv_test) {
    if (1 == exitcode) {
      std::cout << "Passed." << std::endl;
      exitcode = 0;
    } else {
      std::cout << "Failed." << std::endl;
      exitcode = 1;
    }
  } else {
    if (exitcode != 0) {
      std::cout << "*** error: exitcode=" << exitcode << std::endl;
    }
  }  

  return exitcode;
}
