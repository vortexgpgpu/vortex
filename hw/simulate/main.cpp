#include "simulator.h"
#include <iostream>
#include <fstream>
#include <iomanip>

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

inline const char* fileExtension(const char* filepath) {
    const char *ext = strrchr(filepath, '.');
    if (ext == NULL || ext == filepath) 
      return "";
    return ext + 1;
}

int main(int argc, char **argv) {

	int exitcode = 0;
	bool failed = false;
	
	parse_args(argc, argv);

	for (auto program : programs) {
		std::cout << "Running " << program << "..." << std::endl;

		RAM ram;
		Simulator simulator;
		simulator.attach_ram(&ram);

		std::string program_ext(fileExtension(program));
		if (program_ext == "bin") {
			simulator.load_bin(program);
		} else if (program_ext == "hex") {
			simulator.load_ihex(program);
		} else {
			std::cout << "*** error: only *.bin or *.hex images supported." << std::endl;
			return -1;
		}

		exitcode = simulator.run();
		
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
