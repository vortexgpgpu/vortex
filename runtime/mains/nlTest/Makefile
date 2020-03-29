
COMP     = /opt/riscv/bin/riscv32-unknown-elf-gcc
# CC_FLAGS = -march=rv32im -mabi=ilp32 -O0 -Wl,-Bstatic,-T,linker.ld -ffreestanding -nostdlib
CC_FLAGS = -march=rv32im -mabi=ilp32 -O0 -Wl,-Bstatic,-T,../vortex_link.ld -ffreestanding -nostartfiles -nostdlib

DMP  = /opt/riscv/bin/riscv32-unknown-elf-objdump
CPY  = /opt/riscv/bin/riscv32-unknown-elf-objcopy


NEWLIB  = ../../newlib/newlib.c ../../newlib/newlib_notimp.c ../../newlib/newlib.s
VX_STR  = ../../startup/vx_start.S
VX_INT  = ../../intrinsics/vx_intrinsics.s
VX_IO   = ../../io/vx_io.s ../../io/vx_io.c
VX_API  = ../../vx_api/vx_api.c
VX_TEST = ../../tests/tests.c
VX_FIO  = ../../fileio/fileio.s

VX_MAIN = ./vx_nl_main.c

all: HEX DUMP ELF

DUMP: ELF
	$(DMP) -D vx_nl_main.elf > vx_nl_main.dump

HEX: ELF
	$(CPY) -O ihex vx_nl_main.elf vx_nl_main.hex

ELF: 
	$(COMP) $(CC_FLAGS) $(VX_STR) $(VX_FIO) $(NEWLIB) $(VX_INT) $(VX_IO) $(VX_API) $(VX_TEST) $(VX_MAIN) /opt/riscv/riscv32-unknown-elf/lib/libc.a /opt/riscv/riscv32-unknown-elf/lib/libstdc++.a -static-libgcc -lgcc -o vx_nl_main.elf
