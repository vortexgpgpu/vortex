#include "test_bench.h"

#define NUM_TESTS 46

int main(int argc, char **argv)
{

	Verilated::commandArgs(argc, argv);

	Verilated::traceEverOn(true);

	// Verilated::debug(1);


 //    bool passed = true;
	// std::string tests[NUM_TESTS] = {
	// 	"../../emulator/riscv_tests/rv32ui-p-add.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-addi.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-and.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-andi.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-auipc.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-beq.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-bge.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-bgeu.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-blt.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-bltu.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-bne.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-jal.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-jalr.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-lb.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-lbu.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-lh.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-lhu.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-lui.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-lw.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-or.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-ori.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-sb.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-sh.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-simple.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-sll.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-slli.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-slt.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-slti.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-sltiu.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-sltu.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-sra.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-srai.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-srl.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-srli.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-sub.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-sw.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-xor.hex",
	// 	"../../emulator/riscv_tests/rv32ui-p-xori.hex",
	// 	"../../emulator/riscv_tests/rv32um-p-div.hex",
	// 	"../../emulator/riscv_tests/rv32um-p-divu.hex",
	// 	"../../emulator/riscv_tests/rv32um-p-mul.hex",
	// 	"../../emulator/riscv_tests/rv32um-p-mulh.hex",
	// 	"../../emulator/riscv_tests/rv32um-p-mulhsu.hex",
	// 	"../../emulator/riscv_tests/rv32um-p-mulhu.hex",
	// 	"../../emulator/riscv_tests/rv32um-p-rem.hex",
	// 	"../../emulator/riscv_tests/rv32um-p-remu.hex"
	// };

	// 	for (int ii = 0; ii < NUM_TESTS; ii++)
	// 	// for (int ii = 5; ii < 6; ii++)
	// 	{
	// 		std::cout << "TESTING: " << tests[ii] << '\n';
	// 		Vortex v;
	// 		bool curr = v.simulate(tests[ii]);

	// 		if ( curr) std::cerr << GREEN << "Test Passed: " << tests[ii] << std::endl;
	// 		if (!curr) std::cerr << RED   << "Test Failed: " << tests[ii] << std::endl;
	// 		passed = passed && curr;

	// 		std::cerr << DEFAULT;
	// 	}

	// 	if( passed) std::cerr << DEFAULT << "PASSED ALL TESTS\n";
	// 	if(!passed) std::cerr << DEFAULT << "Failed one or more tests\n";


	// char testing[] = "../../emulator/riscv_tests/rv32ui-p-sw.hex";
	Vortex v;
	char testing[] = "../../kernel/vortex_test.hex";

	bool curr = v.simulate(testing);
	if ( curr) std::cerr << GREEN << "Test Passed: " << testing << std::endl;
	if (!curr) std::cerr << RED   << "Test Failed: " << testing << std::endl;

	return 0;

}
