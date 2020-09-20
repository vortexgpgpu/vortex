#include "simulator.h"
#include <iostream>
#include <fstream>
#include <iomanip>

#define ALL_TESTS

int main(int argc, char **argv) {
	if (argc == 1) {
#ifdef ALL_TESTS
	bool passed = true;

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
		"../../../benchmarks/riscv_tests/isa/rv32uf-p-fdiv.hex",
		"../../../benchmarks/riscv_tests/isa/rv32uf-p-fmadd.hex",		
		"../../../benchmarks/riscv_tests/isa/rv32uf-p-fmin.hex",
		"../../../benchmarks/riscv_tests/isa/rv32uf-p-fcmp.hex",		
		"../../../benchmarks/riscv_tests/isa/rv32uf-p-ldst.hex",	 
		"../../../benchmarks/riscv_tests/isa/rv32uf-p-fcvt.hex",
		"../../../benchmarks/riscv_tests/isa/rv32uf-p-fcvt_w.hex",
		"../../../benchmarks/riscv_tests/isa/rv32uf-p-fclass.hex",		
		"../../../benchmarks/riscv_tests/isa/rv32uf-p-move.hex",	
		"../../../benchmarks/riscv_tests/isa/rv32uf-p-recoding.hex",
#endif
	 };

	for (std::string test : tests) {
		std::cerr << DEFAULT << "\n---------------------------------------\n";

		std::cerr << test << std::endl;

		RAM ram;
		Simulator simulator;
		simulator.attach_ram(&ram);
		simulator.load_ihex(test.c_str());
		simulator.run();

		bool status = (1 == simulator.get_last_wb_value(3));

		if (status) std::cerr << GREEN << "Test Passed: " << test << std::endl;
		if (!status) std::cerr << RED   << "Test Failed: " << test << std::endl;
		std::cerr << DEFAULT;
		passed = passed && status;
		if (!passed)
			break;
	}

	for (std::string test : tests_fp) {
		std::cerr << DEFAULT << "\n---------------------------------------\n";

		std::cerr << test << std::endl;

		RAM ram;
		Simulator simulator;
		simulator.attach_ram(&ram);
		simulator.load_ihex(test.c_str());
		simulator.run();

		bool status = (1 == simulator.get_last_wb_value(3));

		if (status) std::cerr << GREEN << "Test Passed: " << test << std::endl;
		if (!status) std::cerr << RED   << "Test Failed: " << test << std::endl;
		std::cerr << DEFAULT;
		passed = passed && status;
		if (!passed)
			break;
	}

	std::cerr << DEFAULT << "\n***************************************\n";

	if (passed) std::cerr << DEFAULT << "PASSED ALL TESTS\n";
	if (!passed) std::cerr << DEFAULT << "Failed one or more tests\n";

	return !passed;

#else

	char test[] = "../../../runtime/tests/simple/vx_simple.hex";

  std::cerr << test << std::endl;

  RAM ram;
	Simulator simulator;
	simulator.attach_ram(&ram);
	simulator.load_ihex(test);
  simulator.run();

  return 0;

#endif

}	else {
	bool passed = true;

	std::vector<std::string> tests(argv+2, argv+argc);
	for (std::string test : tests) {
		std::cerr << DEFAULT << "\n---------------------------------------\n";

		std::cerr << test << std::endl;

		RAM ram;
		Simulator simulator;
		simulator.attach_ram(&ram);
		simulator.load_ihex(test.c_str());
		simulator.run();

		//bool status = (1 == simulator.get_last_wb_value(3));
		bool status = true;
		if (status) std::cerr << GREEN << "Test Passed: " << test << std::endl;
		if (!status) std::cerr << RED   << "Test Failed: " << test << std::endl;
		std::cerr << DEFAULT;
		passed = passed && status;
		if (!passed)
			break;
	}
	return 0;
}

}
