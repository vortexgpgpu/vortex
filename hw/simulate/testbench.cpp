#include "simulator.h"
#include <iostream>
#include <fstream>
#include <iomanip>

//#define ALL_TESTS

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
	bool passed = true;
	if (argc == 1) {
	#ifdef ALL_TESTS
		std::string tests[] = {
			"../../../tests/riscv/isa/rv32ui-p-add.hex",
			"../../../tests/riscv/isa/rv32ui-p-addi.hex",
			"../../../tests/riscv/isa/rv32ui-p-and.hex",
			"../../../tests/riscv/isa/rv32ui-p-andi.hex",
			"../../../tests/riscv/isa/rv32ui-p-auipc.hex",
			"../../../tests/riscv/isa/rv32ui-p-beq.hex",
			"../../../tests/riscv/isa/rv32ui-p-bge.hex",
			"../../../tests/riscv/isa/rv32ui-p-bgeu.hex",
			"../../../tests/riscv/isa/rv32ui-p-blt.hex",
			"../../../tests/riscv/isa/rv32ui-p-bltu.hex",
			"../../../tests/riscv/isa/rv32ui-p-bne.hex",
			"../../../tests/riscv/isa/rv32ui-p-jal.hex",
			"../../../tests/riscv/isa/rv32ui-p-jalr.hex",
			"../../../tests/riscv/isa/rv32ui-p-lb.hex",
			"../../../tests/riscv/isa/rv32ui-p-lbu.hex",
			"../../../tests/riscv/isa/rv32ui-p-lh.hex",
			"../../../tests/riscv/isa/rv32ui-p-lhu.hex",
			"../../../tests/riscv/isa/rv32ui-p-lui.hex",
			"../../../tests/riscv/isa/rv32ui-p-lw.hex",
			"../../../tests/riscv/isa/rv32ui-p-or.hex",
			"../../../tests/riscv/isa/rv32ui-p-ori.hex",
			"../../../tests/riscv/isa/rv32ui-p-sb.hex",
			"../../../tests/riscv/isa/rv32ui-p-sh.hex",
			"../../../tests/riscv/isa/rv32ui-p-simple.hex",
			"../../../tests/riscv/isa/rv32ui-p-sll.hex",
			"../../../tests/riscv/isa/rv32ui-p-slli.hex",
			"../../../tests/riscv/isa/rv32ui-p-slt.hex",
			"../../../tests/riscv/isa/rv32ui-p-slti.hex",
			"../../../tests/riscv/isa/rv32ui-p-sltiu.hex",
			"../../../tests/riscv/isa/rv32ui-p-sltu.hex",
			"../../../tests/riscv/isa/rv32ui-p-sra.hex",
			"../../../tests/riscv/isa/rv32ui-p-srai.hex",
			"../../../tests/riscv/isa/rv32ui-p-srl.hex",
			"../../../tests/riscv/isa/rv32ui-p-srli.hex",
			"../../../tests/riscv/isa/rv32ui-p-sub.hex",
			"../../../tests/riscv/isa/rv32ui-p-sw.hex",
			"../../../tests/riscv/isa/rv32ui-p-xor.hex",
			"../../../tests/riscv/isa/rv32ui-p-xori.hex",
	#ifdef EXT_M_ENABLE
			"../../../tests/riscv/isa/rv32um-p-div.hex",
			"../../../tests/riscv/isa/rv32um-p-divu.hex",
			"../../../tests/riscv/isa/rv32um-p-mul.hex",
			"../../../tests/riscv/isa/rv32um-p-mulh.hex",
			"../../../tests/riscv/isa/rv32um-p-mulhsu.hex",
			"../../../tests/riscv/isa/rv32um-p-mulhu.hex",
			"../../../tests/riscv/isa/rv32um-p-rem.hex",
			"../../../tests/riscv/isa/rv32um-p-remu.hex",
	#endif
		};

		std::string tests_fp[] = {
	#ifdef EXT_F_ENABLE
			"../../../tests/riscv/isa/rv32uf-p-fadd.hex",
			"../../../tests/riscv/isa/rv32uf-p-fmadd.hex",		
			"../../../tests/riscv/isa/rv32uf-p-fmin.hex",
			"../../../tests/riscv/isa/rv32uf-p-fcmp.hex",		
			"../../../tests/riscv/isa/rv32uf-p-ldst.hex",	 
			"../../../tests/riscv/isa/rv32uf-p-fcvt.hex",
			"../../../tests/riscv/isa/rv32uf-p-fcvt_w.hex",	
			"../../../tests/riscv/isa/rv32uf-p-move.hex",	
			"../../../tests/riscv/isa/rv32uf-p-recoding.hex",
			"../../../tests/riscv/isa/rv32uf-p-fdiv.hex",
			"../../../tests/riscv/isa/rv32uf-p-fclass.hex",
	#endif
		};

		for (std::string test : tests) {
			std::cout << "\n***************************************\n";
			std::cout << test << std::endl;

			RAM ram;
			Simulator simulator;
			simulator.attach_ram(&ram);
			simulator.load_ihex(test.c_str());
			int exitcode = simulator.run();

			if (1 == exitcode) {
				std::cout << "Passed" << std::endl;
			} else {
				std::cout << "Failed: exitcode=" << exitcode << std::endl;
				passed = false;
			}

			if (!passed)
				break;
		}

		for (std::string test : tests_fp) {
			std::cout << "\n***************************************\n";
			std::cout << test << std::endl;

			RAM ram;
			Simulator simulator;
			simulator.attach_ram(&ram);
			simulator.load_ihex(test.c_str());
			int exitcode = simulator.run();

			if (1 == exitcode) {
				std::cout << "Passed" << std::endl;
			} else {
				std::cout << "Failed: exitcode=" << exitcode << std::endl;
				passed = false;
			}

			if (!passed)
				break;
		}

		std::cout << "\n***************************************\n";

		if (passed) {
			std::cout << "PASSED ALL TESTS\n";
		} else {
			std::cout << "Failed one or more tests\n";
		}

	#else

		char test[] = "../../../tests/runtime/simple/vx_simple.hex";

		std::cout << test << std::endl;

		RAM ram;
		Simulator simulator;
		simulator.attach_ram(&ram);
		simulator.load_ihex(test);
		int exitcode = simulator.run();

		if (exitcode != 0) {
			std::cout << "*** error: exitcode=" << exitcode << std::endl;
			passed = false;
		}

	#endif

	} else {
		parse_args(argc, argv);

		for (auto program : programs) {
			std::cout << "Running " << program << "..." << std::endl;

			RAM ram;
			Simulator simulator;
			simulator.attach_ram(&ram);
			simulator.load_ihex(program);
			int exitcode = simulator.run();
			
			if (riscv_test) {
				if (1 == exitcode) {
					std::cout << "Passed" << std::endl;
				} else {
					std::cout << "Failed: exitcode=" << exitcode << std::endl;
					passed = false;
				}
			} else {
				if (exitcode != 0) {
					std::cout << "*** error: exitcode=" << exitcode << std::endl;
					passed = false;
				}
			}	
			
			if (!passed)
				break;
		}
	}
	
	return !passed;
}
