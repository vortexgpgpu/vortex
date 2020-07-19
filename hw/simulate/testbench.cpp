#include "simulator.h"
#include <iostream>
#include <fstream>
#include <iomanip>

int main(int argc, char **argv)
{
#define ALL_TESTS
#ifdef ALL_TESTS
	bool passed = true;

  std::string tests[] = {
    "../../../benchmarks/riscv_tests/rv32ui-p-add.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-addi.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-and.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-andi.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-auipc.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-beq.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-bge.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-bgeu.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-blt.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-bltu.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-bne.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-jal.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-jalr.hex",		
	 	"../../../benchmarks/riscv_tests/rv32ui-p-lb.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-lbu.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-lh.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-lhu.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-lui.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-lw.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-or.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-ori.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-sb.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-sh.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-simple.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-sll.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-slli.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-slt.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-slti.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-sltiu.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-sltu.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-sra.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-srai.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-srl.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-srli.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-sub.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-sw.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-xor.hex",
	 	"../../../benchmarks/riscv_tests/rv32ui-p-xori.hex",
	 	"../../../benchmarks/riscv_tests/rv32um-p-div.hex",
	 	"../../../benchmarks/riscv_tests/rv32um-p-divu.hex",
	 	"../../../benchmarks/riscv_tests/rv32um-p-mul.hex",
	 	"../../../benchmarks/riscv_tests/rv32um-p-mulh.hex",
	 	"../../../benchmarks/riscv_tests/rv32um-p-mulhsu.hex",
	 	"../../../benchmarks/riscv_tests/rv32um-p-mulhu.hex",
	 	"../../../benchmarks/riscv_tests/rv32um-p-rem.hex",
	 	"../../../benchmarks/riscv_tests/rv32um-p-remu.hex"
	 };

	for (std::string test : tests) {
		std::cerr << DEFAULT << "\n---------------------------------------\n";

		std::cerr << test << std::endl;

		RAM ram;
		Simulator simulator;
		simulator.attach_ram(&ram);
		simulator.load_ihex(test.c_str());
		bool status = simulator.run();

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
	//char test[] = "../../../benchmarks/riscv_tests/rv32ui-p-lb.hex";
	//char test[] = "../../../benchmarks/riscv_tests/rv32ui-p-lw.hex";
	//char test[] = "../../../benchmarks/riscv_tests/rv32ui-p-sw.hex";

  std::cerr << test << std::endl;

  RAM ram;
	Simulator simulator;
	simulator.attach_ram(&ram);
	simulator.load_ihex(test);
	bool status = simulator.run();

	if (status) std::cerr << GREEN << "Test Passed: " << test << std::endl;
	if (!status) std::cerr << RED   << "Test Failed: " << test << std::endl;

  return !status;

#endif
}