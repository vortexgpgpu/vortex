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
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-add.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-addi.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-and.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-andi.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-auipc.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-beq.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-bge.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-bgeu.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-blt.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-bltu.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-bne.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-jal.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-jalr.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-lb.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-lbu.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-lh.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-lhu.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-lui.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-lw.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-or.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-ori.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-sb.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-sh.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-simple.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-sll.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-slli.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-slt.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-slti.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-sltiu.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-sltu.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-sra.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-srai.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-srl.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-srli.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-sub.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-sw.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-xor.hex",
			"../../../benchmarks/riscv_tests/isa/rv32ui-p-xori.hex",
	#ifdef EXT_M_ENABLE
			"../../../benchmarks/riscv_tests/isa/rv32um-p-div.hex",
			"../../../benchmarks/riscv_tests/isa/rv32um-p-divu.hex",
			"../../../benchmarks/riscv_tests/isa/rv32um-p-mul.hex",
			"../../../benchmarks/riscv_tests/isa/rv32um-p-mulh.hex",
			"../../../benchmarks/riscv_tests/isa/rv32um-p-mulhsu.hex",
			"../../../benchmarks/riscv_tests/isa/rv32um-p-mulhu.hex",
			"../../../benchmarks/riscv_tests/isa/rv32um-p-rem.hex",
			"../../../benchmarks/riscv_tests/isa/rv32um-p-remu.hex",
	#endif
		};

		std::string tests_fp[] = {
	#ifdef EXT_F_ENABLE
			"../../../benchmarks/riscv_tests/isa/rv32uf-p-fadd.hex",
			"../../../benchmarks/riscv_tests/isa/rv32uf-p-fmadd.hex",		
			"../../../benchmarks/riscv_tests/isa/rv32uf-p-fmin.hex",
			"../../../benchmarks/riscv_tests/isa/rv32uf-p-fcmp.hex",		
			"../../../benchmarks/riscv_tests/isa/rv32uf-p-ldst.hex",	 
			"../../../benchmarks/riscv_tests/isa/rv32uf-p-fcvt.hex",
			"../../../benchmarks/riscv_tests/isa/rv32uf-p-fcvt_w.hex",	
			"../../../benchmarks/riscv_tests/isa/rv32uf-p-move.hex",	
			"../../../benchmarks/riscv_tests/isa/rv32uf-p-recoding.hex",
			"../../../benchmarks/riscv_tests/isa/rv32uf-p-fdiv.hex",
			"../../../benchmarks/riscv_tests/isa/rv32uf-p-fclass.hex",
	#endif
		};

		for (std::string test : tests) {
			std::cout << "\n---------------------------------------\n";

			std::cout << test << std::endl;

			RAM ram;
			Simulator simulator;
			simulator.attach_ram(&ram);
			simulator.load_ihex(test.c_str());
			simulator.run();

			bool status = (1 == simulator.get_last_wb_value(3));

			if (status) std::cout << "Passed: " << test << std::endl;
			if (!status) std::cout << "Failed: " << test << std::endl;
			passed = passed && status;
			if (!passed)
				break;
		}

		for (std::string test : tests_fp) {
			std::cout << "\n---------------------------------------\n";

			std::cout << test << std::endl;

			RAM ram;
			Simulator simulator;
			simulator.attach_ram(&ram);
			simulator.load_ihex(test.c_str());
			simulator.run();

			bool status = (1 == simulator.get_last_wb_value(3));

			if (status) std::cout << "Passed: " << test << std::endl;
			if (!status) std::cout << "Failed: " << test << std::endl;
			passed = passed && status;
			if (!passed)
				break;
		}

		std::cout << "\n***************************************\n";

		if (passed) std::cout << "PASSED ALL TESTS\n";
		if (!passed) std::cout << "Failed one or more tests\n";

	#else

		char test[] = "../../../runtime/tests/simple/vx_simple.hex";

		std::cout << test << std::endl;

		RAM ram;
		Simulator simulator;
		simulator.attach_ram(&ram);
		simulator.load_ihex(test);
		simulator.run();

	#endif

	}	else {
		parse_args(argc, argv);

		for (auto program : programs) {
			std::cout << "Running " << program << " .." << std::endl;

			RAM ram;
			Simulator simulator;
			simulator.attach_ram(&ram);
			simulator.load_ihex(program);
			simulator.run();
			
			if (riscv_test) {
				bool status = (1 == simulator.get_last_wb_value(3));
				if (status) std::cout << "Passed." << std::endl;
				if (!status) std::cout << "Failed." << std::endl;		
				passed = passed && status;
				if (!passed)
					break;
			}	
		}
	}
	
	return !passed;
}
