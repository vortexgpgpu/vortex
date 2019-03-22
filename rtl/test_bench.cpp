

#include "test_bench.h"

#define NUM_TESTS 39

int main(int argc, char **argv)
{

	Verilated::commandArgs(argc, argv);


	Vortex v;

    bool passed = true;
	std::string tests[NUM_TESTS] = {
		"../../src/riscv_tests/rv32ui-p-add.hex",
		"../../src/riscv_tests/rv32ui-p-addi.hex",
		"../../src/riscv_tests/rv32ui-p-and.hex",
		"../../src/riscv_tests/rv32ui-p-andi.hex",
		"../../src/riscv_tests/rv32ui-p-auipc.hex",
		"../../src/riscv_tests/rv32ui-p-beq.hex",
		"../../src/riscv_tests/rv32ui-p-bge.hex",
		"../../src/riscv_tests/rv32ui-p-bgeu.hex",
		"../../src/riscv_tests/rv32ui-p-blt.hex",
		"../../src/riscv_tests/rv32ui-p-bltu.hex",
		"../../src/riscv_tests/rv32ui-p-bne.hex",
		"../../src/riscv_tests/rv32ui-p-jal.hex",
		"../../src/riscv_tests/rv32ui-p-jalr.hex",
		"../../src/riscv_tests/rv32ui-p-lb.hex",
		"../../src/riscv_tests/rv32ui-p-lbu.hex",
		"../../src/riscv_tests/rv32ui-p-lh.hex",
		"../../src/riscv_tests/rv32ui-p-lhu.hex",
		"../../src/riscv_tests/rv32ui-p-lui.hex",
		"../../src/riscv_tests/rv32ui-p-lui.hex.hex",
		"../../src/riscv_tests/rv32ui-p-lw.hex",
		"../../src/riscv_tests/rv32ui-p-or.hex",
		"../../src/riscv_tests/rv32ui-p-ori.hex",
		"../../src/riscv_tests/rv32ui-p-sb.hex",
		"../../src/riscv_tests/rv32ui-p-sh.hex",
		"../../src/riscv_tests/rv32ui-p-simple.hex",
		"../../src/riscv_tests/rv32ui-p-sll.hex",
		"../../src/riscv_tests/rv32ui-p-slli.hex",
		"../../src/riscv_tests/rv32ui-p-slt.hex",
		"../../src/riscv_tests/rv32ui-p-slti.hex",
		"../../src/riscv_tests/rv32ui-p-sltiu.hex",
		"../../src/riscv_tests/rv32ui-p-sltu.hex",
		"../../src/riscv_tests/rv32ui-p-sra.hex",
		"../../src/riscv_tests/rv32ui-p-srai.hex",
		"../../src/riscv_tests/rv32ui-p-srl.hex",
		"../../src/riscv_tests/rv32ui-p-srli.hex",
		"../../src/riscv_tests/rv32ui-p-sub.hex",
		"../../src/riscv_tests/rv32ui-p-sw.hex",
		"../../src/riscv_tests/rv32ui-p-xor.hex",
		"../../src/riscv_tests/rv32ui-p-xori.hex",
	};

		for (int ii = 0; ii < NUM_TESTS; ii++)
		// for (int ii = 0; ii < NUM_TESTS - 1; ii++)
		{
			bool curr = v.simulate(tests[ii]);

			if ( curr) std::cout << GREEN << "Test Passed: " << tests[ii] << std::endl;
			if (!curr) std::cout << RED   << "Test Failed: " << tests[ii] << std::endl;
			passed = passed && curr;

			std::cout << DEFAULT;
		}

		if( passed) std::cout << DEFAULT << "PASSED ALL TESTS\n";
		if(!passed) std::cout << DEFAULT << "Failed one or more tests\n";

		
	// v.simulate("../../src/riscv_tests/rv32ui-p-add.hex");

	return 0;

}



