
COMP   = /opt/riscv/bin/riscv32-unknown-elf-gcc
CC_FLAGS = -march=rv32ic -mabi=ilp32 -O0 -Wl,-Bstatic,-T,linker.ld -ffreestanding -nostdlib

DMP  = /opt/riscv/bin/riscv32-unknown-elf-objdump
CPY  = /opt/riscv/bin/riscv32-unknown-elf-objcopy


all: HEX DUMP ELF

DUMP: ELF
	$(DMP) -D gpgpu_test.elf > gpgpu_test.dump

HEX: ELF
	$(CPY) -O ihex gpgpu_test.elf gpgpu_test.hex

ELF: 
	$(COMP) $(CC_FLAGS) ./lib/lib.s gpgpu_test.c ./lib/queue.s ./lib/lib.c -o gpgpu_test.elf
