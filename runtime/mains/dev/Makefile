
COMP     = ../../../../riscv-gnu-toolchain/drops/bin/riscv32-unknown-elf-gcc
# CC_FLAGS = -march=rv32im -mabi=ilp32 -O0 -Wl,-Bstatic,-T,linker.ld -ffreestanding -nostdlib
CC_FLAGS = -march=rv32im -mabi=ilp32 -O0 -Wl,-Bstatic,-T,../vortex_link.ld -ffreestanding -nostartfiles

DMP  = ../../../../riscv-gnu-toolchain/drops/bin/riscv32-unknown-elf-objdump
CPY  = ../../../../riscv-gnu-toolchain/drops/bin/riscv32-unknown-elf-objcopy


VX_STR  = ../../startup/vx_start.s
VX_INT  = ../../intrinsics/vx_intrinsics.s
VX_IO   = ../../io/vx_io.s ../../io/vx_io.c
VX_API  = ../../vx_api/vx_api.c
VX_TEST = ../../tests/tests.c

VX_MAIN = ./vx_dev_main.c

all: HEX DUMP ELF

DUMP: ELF
	$(DMP) -D vx_dev_main.elf > vx_dev_main.dump

HEX: ELF
	$(CPY) -O ihex vx_dev_main.elf vx_dev_main.hex

ELF: 
	$(COMP) $(CC_FLAGS) $(VX_STR) $(VX_INT) $(VX_IO) $(VX_API) $(VX_TEST) $(VX_MAIN)  -o vx_dev_main.elf
