
COMP   = /opt/riscv/bin/riscv32-unknown-elf-gcc
# COMP   = /opt/riscv/bin/riscv32-unknown-linux-gnu-gcc
CC_FLAGS = -march=rv32im -mabi=ilp32 -O0 -Wl,-Bstatic,-T,linker.ld -ffreestanding -nostdlib

DMP  = /opt/riscv/bin/riscv32-unknown-elf-objdump
CPY  = /opt/riscv/bin/riscv32-unknown-elf-objcopy

# DMP  = /opt/riscv/bin/riscv32-unknown-linux-gnu-objdump
# CPY  = /opt/riscv/bin/riscv32-unknown-linux-gnu-objcopy

VX_LIB = ./vx_os/vx_back/vx_back.s ./vx_os/vx_back/vx_back.c ./vx_os/vx_util/queue.s
VX_IO  = ./vx_os/vx_io/vx_io.s ./vx_os/vx_io/vx_io.c
VX_FR  = ./vx_include/vx_front.c

all: HEX DUMP ELF

DUMP: ELF
	$(DMP) -D vortex_test.elf > vortex_test.dump

HEX: ELF
	$(CPY) -O ihex vortex_test.elf vortex_test.hex

ELF: 
	$(COMP) $(CC_FLAGS) $(VX_LIB) $(VX_IO) $(VX_FR) vx_main.c  -o vortex_test.elf
