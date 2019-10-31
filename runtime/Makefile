
COMP     = /opt/riscv/bin/riscv32-unknown-elf-gcc
CC_FLAGS = -march=rv32im -mabi=ilp32 -O0 -Wl,-Bstatic,-T,linker.ld -ffreestanding -nostdlib

DMP  = /opt/riscv/bin/riscv32-unknown-elf-objdump
CPY  = /opt/riscv/bin/riscv32-unknown-elf-objcopy


VX_STR  = ./startup/vx_start.s
VX_INT  = ./intrinsics/vx_intrinsics.s
VX_IO   = ./io/vx_io.s ./io/vx_io.c

VX_MAIN = ./vx_main.c

all: HEX DUMP ELF

DUMP: ELF
	$(DMP) -D vortex_runtime.elf > vortex_runtime.dump

HEX: ELF
	$(CPY) -O ihex vortex_runtime.elf vortex_runtime.hex

ELF: 
	$(COMP) $(CC_FLAGS) $(VX_STR) $(VX_INT) $(VX_IO) $(VX_MAIN)  -o vortex_runtime.elf
