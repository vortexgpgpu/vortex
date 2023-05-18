#include <iostream>
#include <fstream>
#include <iomanip>
#include <unistd.h>
#include <unistd.h>
#include <util.h>
#include <mem.h>
#include <VX_config.h>
#include <VX_types.h>
#include "processor.h"

#define RAM_PAGE_SIZE 4096

using namespace vortex;

static void show_usage() {
   std::cout << "Usage: [-r] [-h: help] programs.." << std::endl;
}

bool riscv_test = false;
std::vector<const char*> programs;

static void parse_args(int argc, char **argv) {
  	int c;
  	while ((c = getopt(argc, argv, "rh?")) != -1) {
    	switch (c) {
		case 'r':
			riscv_test = true;
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

	for (int i = optind; i < argc; ++i) {
		programs.push_back(argv[i]);	
	}
}

int main(int argc, char **argv) {

	int exitcode = 0;
	bool failed = false;
	
	parse_args(argc, argv);

	// create memory module
	vortex::RAM ram(RAM_PAGE_SIZE);

	// create processor
	vortex::Processor processor;

	// attach memory module
	processor.attach_ram(&ram);

	// setup base DCRs
	processor.write_dcr(DCR_BASE_STARTUP_ADDR0, STARTUP_ADDR & 0xffffffff);
#if (XLEN == 64)
    processor.write_dcr(DCR_BASE_STARTUP_ADDR1, STARTUP_ADDR >> 32);
#endif
	processor.write_dcr(DCR_BASE_MPM_CLASS, 0);

	// run simulation
	for (auto program : programs) {
		std::cout << "Running " << program << "..." << std::endl;		

		std::string program_ext(fileExtension(program));
		if (program_ext == "bin") {
			ram.loadBinImage(program, STARTUP_ADDR);
		} else if (program_ext == "hex") {
			ram.loadHexImage(program);
		} else {
			std::cout << "*** error: only *.bin or *.hex images supported." << std::endl;
			return -1;
		}

		exitcode = processor.run();
		
		if (riscv_test) {
			if (1 == exitcode) {
				std::cout << "Passed" << std::endl;
			} else {
				std::cout << "Failed: exitcode=" << exitcode << std::endl;
				failed = true;
			}
		} else {
			if (exitcode != 0) {
				std::cout << "*** error: exitcode=" << exitcode << std::endl;
				failed = true;
			}
		}	
		
		if (failed)
			break;
	}

	return failed ? exitcode : 0;
}
