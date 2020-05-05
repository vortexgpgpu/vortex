#include "simulator.h"
#include <iostream>
#include <fstream>
#include <iomanip>

int main(int argc, char **argv)
{

	// Verilated::debug(1);

	Verilated::commandArgs(argc, argv);

//#define ALL_TESTS
#ifdef ALL_TESTS
	bool passed = true;

  std::string tests[] = {
	 	"../../benchmarks/riscv_tests/rv32ui-p-add.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-addi.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-and.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-andi.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-auipc.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-beq.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-bge.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-bgeu.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-blt.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-bltu.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-bne.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-jal.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-jalr.hex",		
	 	"../../benchmarks/riscv_tests/rv32ui-p-lb.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-lbu.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-lh.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-lhu.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-lui.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-lw.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-or.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-ori.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-sb.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-sh.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-simple.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-sll.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-slli.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-slt.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-slti.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-sltiu.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-sltu.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-sra.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-srai.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-srl.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-srli.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-sub.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-sw.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-xor.hex",
	 	"../../benchmarks/riscv_tests/rv32ui-p-xori.hex",
	 	"../../benchmarks/riscv_tests/rv32um-p-div.hex",
	 	"../../benchmarks/riscv_tests/rv32um-p-divu.hex",
	 	"../../benchmarks/riscv_tests/rv32um-p-mul.hex",
	 	"../../benchmarks/riscv_tests/rv32um-p-mulh.hex",
	 	"../../benchmarks/riscv_tests/rv32um-p-mulhsu.hex",
	 	"../../benchmarks/riscv_tests/rv32um-p-mulhu.hex",
	 	"../../benchmarks/riscv_tests/rv32um-p-rem.hex",
	 	"../../benchmarks/riscv_tests/rv32um-p-remu.hex"
	 };

	for (std::string s : tests) {
		std::cerr << DEFAULT << "\n---------------------------------------\n";

		std::cerr << s << std::endl;

		RAM ram;
		loadHexImpl(s.c_str(), &ram);

		Simulator simulator(&ram);
		bool curr = simulator.run();

		if (curr) std::cerr << GREEN << "Test Passed: " << s << std::endl;
		if (!curr) std::cerr << RED   << "Test Failed: " << s << std::endl;
		std::cerr << DEFAULT;
		passed = passed && curr;
	}

	std::cerr << DEFAULT << "\n***************************************\n";

	if (passed) std::cerr << DEFAULT << "PASSED ALL TESTS\n";
	if(!passed) std::cerr << DEFAULT << "Failed one or more tests\n";

	return !passed;

#else

	char testing[] = "../../runtime/tests/simple/vx_simple_main.hex";
	//char testing[] = "../../benchmarks/riscv_tests/rv32ui-p-lw.hex";
	//char testing[] = "../../benchmarks/riscv_tests/rv32ui-p-sw.hex";

	// const char *testing;

	// if (argc >= 2) {
	//     testing = argv[1];
	// } else {
	//     testing = "../../kernel/vortex_test.hex";
	// }

  std::cerr << testing << std::endl;

  RAM ram;
	loadHexImpl(testing, &ram);

	Simulator simulator(&ram);
	bool curr = simulator.run();

	if (curr) std::cerr << GREEN << "Test Passed: " << testing << std::endl;
	if (!curr) std::cerr << RED   << "Test Failed: " << testing << std::endl;

    return !curr;

#endif
}