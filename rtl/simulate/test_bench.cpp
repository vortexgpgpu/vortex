#include "Vortex.h"

#define NUM_TESTS 46

int main(int argc, char **argv)
{

	// Verilated::debug(1);

	Verilated::commandArgs(argc, argv);

// #define ALL_TESTS
#ifdef ALL_TESTS
     bool passed = true;

  std::string tests[NUM_TESTS] = {
	 	"../../emulator/riscv_tests/rv32ui-p-add.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-addi.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-and.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-andi.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-auipc.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-beq.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-bge.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-bgeu.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-blt.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-bltu.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-bne.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-jal.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-jalr.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-lb.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-lbu.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-lh.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-lhu.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-lui.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-lw.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-or.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-ori.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-sb.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-sh.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-simple.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-sll.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-slli.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-slt.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-slti.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-sltiu.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-sltu.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-sra.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-srai.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-srl.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-srli.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-sub.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-sw.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-xor.hex",
	 	"../../emulator/riscv_tests/rv32ui-p-xori.hex",
	 	"../../emulator/riscv_tests/rv32um-p-div.hex",
	 	"../../emulator/riscv_tests/rv32um-p-divu.hex",
	 	"../../emulator/riscv_tests/rv32um-p-mul.hex",
	 	"../../emulator/riscv_tests/rv32um-p-mulh.hex",
	 	"../../emulator/riscv_tests/rv32um-p-mulhsu.hex",
	 	"../../emulator/riscv_tests/rv32um-p-mulhu.hex",
	 	"../../emulator/riscv_tests/rv32um-p-rem.hex",
	 	"../../emulator/riscv_tests/rv32um-p-remu.hex"
	 };

    for (std::string s : tests) {
        std::cerr << DEFAULT << "\n---------------------------------------\n";

        std::cerr << s << std::endl;

        RAM ram;
	      loadHexImpl(s.c_str(), &ram);

				Vortex v(&ram);
        bool curr = v.simulate();

        if ( curr) std::cerr << GREEN << "Test Passed: " << s << std::endl;
        if (!curr) std::cerr << RED   << "Test Failed: " << s << std::endl;
        std::cerr << DEFAULT;
        passed = passed && curr;
    }

    std::cerr << DEFAULT << "\n***************************************\n";

    if( passed) std::cerr << DEFAULT << "PASSED ALL TESTS\n";
	if(!passed) std::cerr << DEFAULT << "Failed one or more tests\n";

	return !passed;

	#else

	char testing[] = "../../runtime/mains/simple/vx_simple_main.hex";
	//char testing[] = "../../emulator/riscv_tests/rv32ui-p-lw.hex";
	//char testing[] = "../../emulator/riscv_tests/rv32ui-p-sw.hex";

	// const char *testing;

	// if (argc >= 2) {
	//     testing = argv[1];
	// } else {
	//     testing = "../../kernel/vortex_test.hex";
	// }

  std::cerr << testing << std::endl;

  RAM ram;
	loadHexImpl(testing, &ram);

	Vortex v(&ram);
	bool curr = v.simulate();

	if ( curr) std::cerr << GREEN << "Test Passed: " << testing << std::endl;
	if (!curr) std::cerr << RED   << "Test Failed: " << testing << std::endl;

    return !curr;

#endif
}